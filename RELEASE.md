# Releasing torch2casadi

The release workflow publishes the exact wheel and source archive built and tested by CI. It does not rebuild between testing, attestation and upload. Publication is manually dispatched from a version tag; pushing to main does not publish.

## One-time setup

Create pending Trusted Publishers at both:

- https://pypi.org/manage/account/publishing/
- https://test.pypi.org/manage/account/publishing/

Use these fields (environment differs between the two services):

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| PyPI project name | torch2casadi | torch2casadi |
| GitHub owner | casadi | casadi |
| Repository | torch2casadi | torch2casadi |
| Workflow filename | release.yml | release.yml |
| Environment | pypi | testpypi |

A pending publisher creates the project on the first upload. It does not reserve the name. No API token needs to be stored in GitHub.

GitHub environments `pypi` and `testpypi` must match those names. A required reviewer can be configured on `pypi` if desired.

## Release procedure

1. Update the version in `pyproject.toml` and `CHANGELOG.md`, commit, and wait for CI.
2. Tag that commit with `v` followed by the exact version (initial candidate: `v0.1`). Push the tag.
3. Dispatch `.github/workflows/release.yml` from that tag with target `testpypi`.
4. Inspect the TestPyPI files, metadata, provenance and install behavior.
5. Dispatch the same workflow/tag with target `pypi` when ready to publish.

Example dispatch (uploads real files; run only when ready):

```sh
gh workflow run release.yml --repo casadi/torch2casadi --ref v0.1 -f target=testpypi
```

The workflow verifies tag/version agreement, runs CI against built wheels on Python 3.10–3.12, creates GitHub build-provenance attestations, and uploads using OIDC Trusted Publishing with PyPI attestations enabled. All actions are pinned to commit hashes.

## Verifying provenance

For a downloaded release artifact:

```sh
gh attestation verify torch2casadi-0.1-py3-none-any.whl --repo casadi/torch2casadi
```

PyPI's package-file details expose the publishing attestations. They can also be checked with the `pypi-attestations` tool and the artifact's direct PyPI download URL:

```sh
pypi-attestations verify pypi --repository https://github.com/casadi/torch2casadi "$WHEEL_URL"
```

GitHub's build attestation records the source/workflow identity and artifact digest. PyPI's publish attestation binds the uploaded distribution to the Trusted Publisher. They establish provenance, not proof that the package is bug-free.

References: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/), [PyPI attestations](https://docs.pypi.org/attestations/producing-attestations/), [GitHub build provenance](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations).
