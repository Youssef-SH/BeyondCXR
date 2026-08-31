# Local data workspace

Raw sources and generated patient-level artifacts belong here and remain outside Git.

| Location | Contents |
| --- | --- |
| `raw/rsna/extracted/` | RSNA Stage 2 source release |
| `raw/symile/extracted/` | Operator-supplied Symile-MIMIC release |
| `manifests/rsna/bundles/` | Validated RSNA bundles, including split assignments |
| `manifests/symile/bundles/` | Validated Symile bundles |
| `manifests/symile/cv/` | Separate development CV assignments |
| `cache/rsna/` | Disposable deterministic CXR caches |

Source roots can be configured during setup. See the [RSNA](../docs/datasets/rsna.md) and
[Symile](../docs/datasets/symile.md) source guides for release layouts and qualification, and the
[data contract](../docs/data_contract.md) for artifact schemas. Access and transfer constraints
are described in the [data statement](../docs/data_statement.md).
