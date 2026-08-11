"""hrsim4 -- rebuilt real-resume screening corpus.

Fixes every defect the 2026-08-04 audit of hrsim_v3 found in the v1/v2/v3
corpus. The v1-v3 pipeline flattened LiveCareer's `Resume_str` column into a
wall of wrapped prose and truncated it at 2,800 characters mid-sentence; this
package parses the `Resume_html` column instead, which carries the real
document structure (named sections, job entries with company/title/dates,
bullet lists, two-column skill blocks), and lays it out on US Letter pages.
"""
