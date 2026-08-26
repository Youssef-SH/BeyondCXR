# Data statement

## Scope and dataset roles

Symile-MIMIC is the primary multimodal study. RSNA Stage 2 supplies the supporting DICOM,
image-modeling, metadata-fusion, and localization foundation. Their endpoints differ, and RSNA is
not external validation of the Symile predictor.

## Symile-MIMIC

BeyondCXR uses Symile-MIMIC 1.0.0 with MIMIC-CXR-JPG 2.0.0, MIMIC-IV 2.2, and MIMIC-IV-ECG 1.0.

Access to Symile-MIMIC and the credentialed MIMIC parent resources requires PhysioNet
credentialing, CITI Data or Specimens Only Research training, acceptance of the applicable
PhysioNet Credentialed Health Data License 1.5.0, and the corresponding data-use agreement.
Individual parent resources retain their own published access and licensing terms.

Symile-MIMIC was created for multimodal representation learning and zero-shot CXR retrieval.
BeyondCXR uses its synchronized admission cohort for supervised prediction of an explicit
radiology-derived Pneumonia finding.

### Modalities and timing

- CXR: earliest eligible AP or PA image 24–72 hours after admission;
- laboratories: earliest values for 50 selected blood tests within 24 hours;
- ECG: earliest valid 12-lead recording within 24 hours;
- eligibility: all three modalities and at least one observed laboratory value.

The authenticated source records permit direct verification of represented CXR, ECG, alignment,
and view facts.

Laboratory event timing is authenticated through the official source contract because raw parent
event tables are outside the authenticated bundle scope.

### Cohort, split, and endpoint

The source index contains 11,622 admissions. Official classification membership contains 10,000
training, 750 validation, and 464 unique test-query admissions: 11,214 total. The remaining 408
admissions are audited and excluded. Retrieval-negative candidates and `val_retrieval.csv` are not
classification samples.

The primary endpoint is `pneumonia_strict`: `Pneumonia = 1` is positive, `Pneumonia = 0` is
negative, and uncertain or missing labels are excluded. Official splits are patient-disjoint.

Development combines eligible official train and validation admissions; test remains closed until
the validated pre-test freeze and same-freeze opening record exist.

The strict-label analytic cohort is fixed before model evaluation:

| Scope | Admissions | Positive | Negative |
| --- | ---: | ---: | ---: |
| Train | 2,194 | 1,026 | 1,168 |
| Validation | 174 | 78 | 96 |
| Development | 2,368 | 1,104 | 1,264 |
| Held-out test | 110 | 58 | 52 |

Development is the eligible official train and validation membership; the total strict eligible
cohort is 2,478 admissions. These are cohort and eligibility facts, not model-performance results.

### Laboratories and source authentication

The source representation contains 50 laboratory values and corresponding observedness indicators.
Missing values remain distinct from observed finite measurements. Model-side transformation and
imputation are documented in [training](training.md).

Bundle construction verifies `SHA256SUMS.txt`, required CSV and NPY resources, schemas, dtypes,
shapes, identifier alignment, split isolation, and modality contracts. Large arrays remain external;
the immutable lean bundle records authenticated references and typed cohort data.

## RSNA Stage 2

The labeled training release contains 26,684 DICOM images, 6,012 positive samples, and 9,555
bounding boxes. BeyondCXR uses the challenge target and boxes for supporting CXR qualification.

The target is a challenge annotation, not confirmed infectious pneumonia. Source files remain
under competition terms; the bundle authenticates each labeled DICOM.

## Privacy, transfer, and distribution

Raw data, patient-level bundles, source inventories, predictions, real examples, and private error
review remain outside version control. Public outputs are aggregate. Dataset agreements govern
transfer independently of the code license, and source access does not establish permission to
redistribute trained packages.

## Sources and citations

- Saporta, Adriel; Puli, Aahlad Manas; Goldstein, Mark; and Ranganath, Rajesh.
  *Symile-MIMIC: a multimodal clinical dataset of chest X-rays, electrocardiograms, and blood labs
  from MIMIC-IV.* Version 1.0.0. PhysioNet, 2025. DOI:
  [10.13026/3vvj-s428](https://doi.org/10.13026/3vvj-s428).

- Saporta, Adriel; Puli, Aahlad Manas; Goldstein, Mark; and Ranganath, Rajesh. “Contrasting with
  Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities.” *Advances in
  Neural Information Processing Systems* 37 (NeurIPS 2024). DOI:
  [10.52202/079017-1814](https://doi.org/10.52202/079017-1814).

- Pollard, Tom; Moody, Benjamin E.; Lehman, Li-wei H.; Gow, Brian J.; Fernandes, Chrystinne; Xie,
  Chen; Johnson, Alistair; Mark, Roger G.; and Heldt, Thomas. “PhysioNet as a global platform for
  biomedical research.” *Nature Health* 1, 792–795 (2026). DOI:
  [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

- Johnson, Alistair; Bulgarelli, Lucas; Pollard, Tom; Horng, Steven; Celi, Leo Anthony; and Mark,
  Roger. *MIMIC-IV.* Version 2.2. PhysioNet, 2023. DOI:
  [10.13026/6mm1-ek67](https://doi.org/10.13026/6mm1-ek67).

- Johnson, Alistair E. W.; Bulgarelli, Lucas; Shen, Lu; et al. “MIMIC-IV, a freely accessible
  electronic health record dataset.” *Scientific Data* 10, 1 (2023). DOI:
  [10.1038/s41597-022-01899-x](https://doi.org/10.1038/s41597-022-01899-x).

- Gow, Brian; Pollard, Tom; Nathanson, Larry A.; Johnson, Alistair; Moody, Benjamin; Fernandes,
  Chrystinne; Greenbaum, Nathaniel; Waks, Jonathan W.; Eslami, Parastou; Carbonati, Tanner;
  Chaudhari, Ashish; Herbst, Elizabeth; Moukheiber, Dana; Berkowitz, Seth; Mark, Roger; and Horng,
  Steven. *MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset.* Version 1.0. PhysioNet,
  2023. DOI: [10.13026/4nqg-sb35](https://doi.org/10.13026/4nqg-sb35).

- Johnson, Alistair; Lungren, Matt; Peng, Yifan; Lu, Zhiyong; Mark, Roger; Berkowitz, Seth; and
  Horng, Steven. *MIMIC-CXR-JPG — chest radiographs with structured labels.* Version 2.0.0.
  PhysioNet, 2019. DOI:
  [10.13026/8360-t248](https://doi.org/10.13026/8360-t248).

- Johnson, Alistair E. W.; Pollard, Tom J.; Greenbaum, Nathaniel R.; Lungren, Matthew P.; Deng,
  Chih-ying; Peng, Yifan; Lu, Zhiyong; Mark, Roger G.; Berkowitz, Seth J.; and Horng, Steven.
  “MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs.”
  arXiv:1901.07042 (2019).

- Johnson, Alistair; Pollard, Tom; Mark, Roger; Berkowitz, Seth; and Horng, Steven.
  *MIMIC-CXR Database.* Version 2.0.0. PhysioNet, 2019. DOI:
  [10.13026/C2JT1Q](https://doi.org/10.13026/C2JT1Q).

- Johnson, Alistair E. W.; Pollard, Tom J.; Berkowitz, Seth J.; Greenbaum, Nathaniel R.; Lungren,
  Matthew P.; Deng, Chih-ying; Mark, Roger G.; and Horng, Steven. “MIMIC-CXR, a de-identified
  publicly available database of chest radiographs with free-text reports.” *Scientific Data* 6,
  317 (2019). DOI:
  [10.1038/s41597-019-0322-0](https://doi.org/10.1038/s41597-019-0322-0).

- [RSNA Pneumonia Detection Challenge (2018)](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

- [RSNA Pneumonia Detection Challenge — Terms of Use and Attribution](https://www.rsna.org/-/media/Files/RSNA/Education/AI-resources-and-training/AI-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.ashx).

- Shih, George; Wu, Carol C.; Halabi, Safwan S.; et al. “Augmenting the National Institutes of
  Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia.”
  *Radiology: Artificial Intelligence* 1(1), e180041 (2019). DOI:
  [10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041).

- Wang, Xiaosong; Peng, Yifan; Lu, Le; Lu, Zhiyong; Bagheri, Mohammadhadi; and Summers, Ronald M.
  “ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised
  Classification and Localization of Common Thorax Diseases.” IEEE Conference on Computer Vision
  and Pattern Recognition (CVPR), 3462–3471 (2017). DOI:
  [10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369).

- [NIH Clinical Center Chest X-ray parent-dataset download](https://nihcc.app.box.com/v/ChestXray-NIHCC).

The RSNA Pneumonia Detection Challenge data derive from the NIH Clinical Center Chest X-ray
dataset; the NIH Clinical Center is acknowledged as the parent data provider. The pinned
source-release pages and attribution records define the applicable source citations, licenses,
access conditions, and attribution requirements.

The endpoint is report-derived; this retrospective study does not establish diagnosis, causality,
prospective effectiveness, device readiness, or clinical authorization.
