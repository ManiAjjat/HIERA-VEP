# Public data reconstruction

No bulk research dataset is distributed in this repository. The small examples reproduce the existing public Step 56B example only.

Public regulatory assay records originate from [MaveDB](https://www.mavedb.org/). Step 32 inventories regulatory score sets and records accession IDs; Step 33 downloads and audits the selected public records; Step 34 harmonizes variant and sequence records. Reconstruct accession manifests and retain accession versions and assay metadata. Earlier within-element analyses additionally require their original upstream harmonized training tables, which are not bundled. A complete raw-to-result run is not claimed from the public checkout alone.

Annotation sources are [ENCODE](https://www.encodeproject.org/), the [UCSC Genome Browser](https://genome.ucsc.edu/), [GENCODE](https://www.gencodegenes.org/), [JASPAR](https://jaspar.elixir.no/), and [GTEx](https://gtexportal.org/). See Steps 49A–49C, 29B, 53A and 53B for coordinate eligibility, genome builds, resource URLs, matching and annotation rules. Use the recorded GRCh38 build where required. Respect dataset licenses and access conditions; controlled data must not be uploaded.

The optional frozen Nucleotide Transformer comparison fetches its public pretrained model using the identifier in Steps 36 and 51. Pretrained weights are not included or relicensed. Download only when reproducing that optional comparison.

Excluded categories: raw MaveDB downloads; full variant/intermediate tables; controlled/private data; GTEx/API query caches; raw ENCODE files; genomic FASTA/BAM/CRAM/VCF/BED/HDF5/Parquet archives; foundation-model weights; bootstrap replicate matrices; temporary renders/predictions; environments and caches; manuscript drafts, correspondence and reviewer responses; personal information; credentials; Streamlit deployment assets.
