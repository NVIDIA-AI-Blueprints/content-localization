# scripts/nims

Helpers for the NVIDIA NIM containers we orchestrate (Active Speaker Detection,
LipSync) plus the cross-NIM parity test driver.

| Script | Purpose |
| --- | --- |
| `deploy_asd.sh` | Deploy the Active Speaker Detection NIM. |
| `deploy_lipsync.sh` | Deploy the LipSync NIM. |
| `parity_test.sh` | Run Controller, Direct, and Batch clients on every video and compare outputs. |
