# Copyright 2017 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""A VariantCaller producing DeepVariantCall and gVCF records."""

import abc
import functools
import math
import statistics
from typing import Dict, Sequence, Tuple

import numpy as np

from deepvariant.protos import deepvariant_pb2
from deepvariant.python import allelecounter
from deepvariant.python import variant_calling
from deepvariant.python import variant_calling_multisample
from third_party.nucleus.protos import variants_pb2
from third_party.nucleus.util import genomics_math
from third_party.nucleus.util import ranges
from third_party.nucleus.util import variantcall_utils
from third_party.nucleus.util import vcf_constants


# Reference bases with genotype calls must be one of these four values.
CANONICAL_DNA_BASES = frozenset('ACGT')

# Possible DNA base codes seen in a reference genome.
EXTENDED_IUPAC_CODES = frozenset('ACGTRYSWKMBDHVN')

LOG_10 = math.log(10.0)

IMPOSSIBLE_PROBABILITY_LOG10 = 999.0


def _rescale_read_counts_if_necessary(
    n_ref_reads, n_total_reads, max_allowed_reads
):
  """Ensures that n_total_reads <= max_allowed_reads, rescaling if necessary.

  This function ensures that n_total_reads <= max_allowed_reads. If
  n_total_reads is <= max_allowed_reads, n_ref_reads and n_total_reads are just
  returned. However, if n_total_reads > max_allowed_reads, then n_ref_reads and
  n_total_reads are rescaled to new values n_ref_reads' and n_total_reads' so
  that n_total_reads' == max_allowed_reads and n_ref_reads' / n_total_reads' ~
  n_ref_reads / n_total_reads.

  Args:
    n_ref_reads: int. Number of reference supporting reads.
    n_total_reads: int. Total number of reads.
    max_allowed_reads: int. The maximum value allowed for n_total after
      rescaling, if necessary.

  Returns:
    New values for n_ref_reads and n_total_reads.
  """
  if n_total_reads > max_allowed_reads:
    ratio = n_ref_reads / (1.0 * n_total_reads)
    n_ref_reads = int(math.ceil(ratio * max_allowed_reads))
    n_total_reads = max_allowed_reads
  return n_ref_reads, n_total_reads


def _quantize_gq(raw_gq, binsize):
  """Returns a quantized value of GQ in units of binsize.

  Args:
    raw_gq: int. The raw GQ value to quantize.
    binsize: positive int. The size of bins to quantize within.

  Returns:
    A quantized GQ integer.
  """
  if raw_gq < 1:
    return 0
  else:
    bin_number = (raw_gq - 1) // binsize
    return bin_number * binsize + 1


class VariantCaller(metaclass=abc.ABCMeta):
  """BaseClass for variant callers."""

  def __init__(self, options, use_cache_table, max_cache_coverage):
    self.options = options
    self.cpp_variant_caller = variant_calling_multisample.VariantCaller(
        self.options
    )
    self.cpp_variant_caller_from_vcf = variant_calling.VariantCaller(
        self.options
    )

    self.par_regions = None
    if self.options.par_regions_bed:
      self.par_regions = ranges.RangeSet.from_bed(
          self.options.par_regions_bed, enable_logging=False
      )

    self.max_cache_coverage = max_cache_coverage
    # pylint: disable=g-complex-comprehension
    if use_cache_table:
      self.table = [
          [
              [
                  self._calc_reference_confidence(n_ref, n_total, is_haploid)
                  for n_ref in range(n_total + 1)
              ]
              for n_total in range(self.max_cache_coverage + 1)
          ]
          for is_haploid in [False, True]
      ]
    else:
      self.table = None
    # pylint: enable=g-complex-comprehension

  def reference_confidence(self, n_ref, n_total, is_haploid=False):
    """Computes the confidence that a site in the genome has no variation.

    Computes this confidence using only the counts of the number of reads
    supporting the reference allele and the total number of reads at this site.

    See: https:www.broadinstitute.org/gatk/guide/article?id=4017 for
    background.

    Computes the reference confidence for site allele_count.

    Examines the number of reference supporting and alternate supporting reads
    in allele_count and estimates the genotype likelihoods and confidence that
    this site is homozygous reference. These values are written into the first
    VariantCall record of variant, into the repeated field genotype_likelihood
    and the map field GQ.

    The genotype likelihoods are computed against any possible alternative
    allele, the so-called <*> allele, which boils down to a model that looks
    like:

     log10_p_ref = (1 - p_error)^(ref_n) (p_error)^(non_ref_n)
     log10_p_het = (0.5)^(total_n)
     log10_p_hom_alt = (p_e)^(ref_n) (1 - p_error)^(non_ref_n)

    ref_n is the number of reference supporting reads and non_ref_n is the sum
    of any reads supporting any alternate alleles. Non-informative reads are
    excluded from the calculation.

    and written in as the normalized log10 values so that:

     sum(10^genotype_likelihoods) = 1

    The GQ, according to the VCF specification, is the conditional genotype
    quality, encoded as a phred quality -10 * log10 p(genotype call is wrong,
    conditioned on the site's being variant, as an integer. See:
    https:samtools.github.io/hts-specs/VCFv4.3.pdf
    We are calculating the GQ not for the best genotype, but the GQ of the 0/0
    genotype, regardless of the likelihoods.
    1 = pRR + pRA + pAA; [R is reference, A=<*> is any alternative alternative]
    GQ of 0/0 = -10 * log10(pRA + pAA) [prob that any other differen genotype]
              = -10 * log10(1 - pRR) [substitution from the previous equation]
    Here we don't have pRR directly, but rather log10(pRR).

    Args:
      n_ref: int >= 0 and <= n_total: The number of reads supporting the
        reference allele.
      n_total: int >= 0 and >= n_ref: The number of reads supporting any allele
        at this site.
      is_haploid: bool. If True, the position should be haploid.

    Returns:
      A tuple of two values. The first is an integer value for the GQ (genotype
      quality) and the second is an array-like of the log10 probabilities for
      each of the three genotype configurations.
    """
    if self.table is None:
      return self._calc_reference_confidence(n_ref, n_total, is_haploid)
    else:
      ref_index, total_index = _rescale_read_counts_if_necessary(
          n_ref, n_total, self.max_cache_coverage
      )
      return self.table[is_haploid][total_index][ref_index]

  def _calc_reference_confidence(self, n_ref, n_total, is_haploid=False):
    """Performs the calculation described in reference_confidence()."""
    if n_ref < 0:
      raise ValueError('n_ref={} must be >= 0'.format(n_ref))
    if n_total < n_ref:
      raise ValueError('n_total={} must be >= n_ref={}'.format(n_total, n_ref))
    if self.options.ploidy != 2:
      raise ValueError(
          'ploidy={} but we only support ploidy=2'.format(self.options.ploidy)
      )
    if n_total == 0:
      if is_haploid:
        # No coverage case - all likelihoods are log10 of 1/2, 0, 1/2.
        log10_probs = genomics_math.normalize_log10_probs(
            [-1.0, -IMPOSSIBLE_PROBABILITY_LOG10, -1.0]
        )
      else:
        # No coverage case - all likelihoods are log10 of 1/3, 1/3, 1/3.
        log10_probs = genomics_math.normalize_log10_probs([-1.0, -1.0, -1.0])
    else:
      n_alts = n_total - n_ref
      logp = math.log(self.options.p_error) / LOG_10
      log1p = math.log1p(-self.options.p_error) / LOG_10
      log10_p_ref = n_ref * log1p + n_alts * logp
      log10_p_het = -n_total * math.log(self.options.ploidy) / LOG_10
      if is_haploid:
        log10_p_het = -IMPOSSIBLE_PROBABILITY_LOG10
      log10_p_hom_alt = n_ref * logp + n_alts * log1p
      log10_probs = genomics_math.normalize_log10_probs(
          [log10_p_ref, log10_p_het, log10_p_hom_alt]
      )

    gq = genomics_math.log10_ptrue_to_phred(log10_probs[0], self.options.max_gq)
    gq = int(min(np.floor(gq), self.options.max_gq))
    return gq, log10_probs

  def make_gvcfs(self, allele_count_summaries, include_med_dp=False):
    """Primary interface function for computing gVCF confidence at a site.

    Looks at the counts in the provided list of AlleleCountSummary protos and
    returns properly-formatted Variant protos containing gVCF reference
    blocks for all sites in allele_count_summaries. The returned Variant has
    reference_name, start, end are set and contains a single VariantCall in the
    calls field with call_set_name of options.sample_name, genotypes set to 0/0
    (diploid reference), a GQ value bound in the info field appropriate to the
    data in allele_count, and a MIN_DP value which is the minimum read coverage
    seen in the block.

    The provided allele count must have either a canonical DNA sequence base (
    A, C, G, T) or be "N".

    Args:
      allele_count_summaries: iterable of AlleleCountSummary protos in
        coordinate-sorted order. Each proto is used to get the read counts for
        reference and alternate alleles, the reference position, and reference
        base.
      include_med_dp: boolean. If True, in the gVCF records, we will include
        MED_DP.

    Yields:
      third_party.nucleus.protos.Variant proto in
      coordinate-sorted order containing gVCF records.
    """

    par_regions = None
    if self.options.par_regions_bed:
      par_regions = ranges.RangeSet.from_bed(
          self.options.par_regions_bed, enable_logging=False
      )

    haploid_contigs = frozenset(self.options.haploid_contigs)
    gq_resolution = self.options.gq_resolution

    # Nearby positions frequently have identical read counts. Cache both the
    # confidence lookup and its derived block metadata, keeping the cache local
    # so callers can change options between regions. Bound its size even for
    # unusually high or variable coverage.
    @functools.lru_cache(maxsize=512)
    def confidence(n_ref, n_total, is_haploid):
      raw_gq, likelihoods = self.reference_confidence(
          n_ref, n_total, is_haploid
      )
      quantized_gq = _quantize_gq(raw_gq, gq_resolution)
      has_valid_gl = (
          likelihoods[0] >= likelihoods[1] and likelihoods[0] >= likelihoods[2]
      )
      return quantized_gq, raw_gq, likelihoods, has_valid_gl

    def make_gvcf(
        first, last, raw_gq, likelihoods, min_dp, depths, called=True
    ):
      call = variants_pb2.VariantCall(
          call_set_name=self.options.sample_name,
          genotype=[0, 0] if called else [-1, -1],
          genotype_likelihood=likelihoods,
      )
      variantcall_utils.set_gq(call, raw_gq)
      variantcall_utils.set_min_dp(call, min_dp)
      if include_med_dp:
        med_dp = int(statistics.median(depths)) if called else min_dp
        variantcall_utils.set_med_dp(call, med_dp)
      return variants_pb2.Variant(
          reference_name=first.reference_name,
          reference_bases=first.ref_base,
          alternate_bases=[vcf_constants.GVCF_ALT_ALLELE],
          start=first.position,
          end=last.position + 1,
          calls=[call],
      )

    # Aggregate each block in one pass, without allocating a record for every
    # reference base. Only retain per-base depths when MED_DP is requested.
    first_record = None
    for summary_counts in allele_count_summaries:
      ref_base = summary_counts.ref_base
      if ref_base in CANONICAL_DNA_BASES:
        n_total = summary_counts.total_read_count
        is_haploid = False
        if haploid_contigs and summary_counts.reference_name in haploid_contigs:
          is_haploid = not (
              par_regions
              and par_regions.overlaps(
                  summary_counts.reference_name, summary_counts.position
              )
          )
        quantized_gq, raw_gq, likelihoods, has_valid_gl = confidence(
            summary_counts.ref_supporting_read_count, n_total, is_haploid
        )
      elif ref_base in EXTENDED_IUPAC_CODES:
        # An ambiguous reference base ends the preceding block and is skipped.
        quantized_gq, has_valid_gl = None, True
      else:
        raise ValueError(
            'Invalid reference base={} found during gvcf calculation'.format(
                ref_base
            )
        )

      if first_record is not None and (
          quantized_gq != block_gq or not has_valid_gl
      ):
        yield make_gvcf(
            first_record, last_record, min_gq, min_likelihoods, min_dp, depths
        )
        first_record = None

      if quantized_gq is None:
        continue
      if not has_valid_gl:
        # Sites whose hom-ref likelihood is not maximal remain separate ./.
        # records, even when adjacent sites have the same GQ.
        yield make_gvcf(
            summary_counts,
            summary_counts,
            raw_gq,
            likelihoods,
            n_total,
            None,
            called=False,
        )
        continue

      if first_record is None:
        first_record = summary_counts
        block_gq = quantized_gq
        min_gq, min_likelihoods, min_dp = raw_gq, likelihoods, n_total
        depths = [] if include_med_dp else None
      else:
        # Preserve the first record's likelihoods when raw GQs are tied.
        if raw_gq < min_gq:
          min_gq, min_likelihoods = raw_gq, likelihoods
        if n_total < min_dp:
          min_dp = n_total
      last_record = summary_counts
      if include_med_dp:
        depths.append(n_total)

    if first_record is not None:
      yield make_gvcf(
          first_record, last_record, min_gq, min_likelihoods, min_dp, depths
      )

  def calls_and_gvcfs(
      self,
      allele_counters: Dict[str, allelecounter.AlleleCounter],
      target_sample: str,
      target_role: str = '',
      include_gvcfs: bool = False,
      include_med_dp: bool = False,
      left_padding: int = 0,
      right_padding: int = 0,
  ) -> Tuple[
      Sequence[deepvariant_pb2.DeepVariantCall], Sequence[variants_pb2.Variant]
  ]:
    """Gets variant calls and gvcf records for all sites in allele_counter.

    Args:
      allele_counters: Dictionary of AlleleCounter objects keyed by sample IDs
        holding the allele counts we will use to find candidate variants and
        create gvcf records.
      target_sample: string. Sample ID of sample for which variants are called.
      target_role: string. Role of the target sample.
      include_gvcfs: boolean. If True, we will compute gVCF records for all of
        the AlleleCounts in AlleleCounter.
      include_med_dp: boolean. If True, in the gVCF records, we will include
        MED_DP.
      left_padding: int. Left padding that is added to the region and needs to
        be discarded for candidates and gvcf calculation.
      right_padding: int. Right padding that is added to the region and needs to
        be discarded for candidates and gvcf calculation.

    Returns:
      Two values. The first is a list of DeepVariantCall protos containing our
      candidate variants. The second is a list of gVCF blocks in Variant proto
      format, if include_gvcfs is True. If False, an empty list is returned.
    """

    # TODO Consider passing left and right padding to get_candidates so
    # that we didn't waiste runtime on calculating candidates beoynd the region.
    candidates = self.get_candidates(
        allele_counters=allele_counters,
        sample_name=target_sample,
        sample_role=target_role,
    )

    gvcfs = []
    if include_gvcfs:
      gvcfs = list(
          self.make_gvcfs(
              allele_counters[target_sample].summary_counts(
                  left_padding, right_padding
              ),
              include_med_dp=include_med_dp,
          )
      )
    return candidates, gvcfs

  @abc.abstractmethod
  def get_candidates(
      self,
      allele_counters: Dict[str, allelecounter.AlleleCounter],
      sample_name: str,
      sample_role: str = '',
  ):
    raise NotImplementedError

  @abc.abstractmethod
  def get_candidate_positions(
      self,
      allele_counters: Dict[str, allelecounter.AlleleCounter],
      sample_name: str,
      sample_role: str = '',
  ):
    raise NotImplementedError
