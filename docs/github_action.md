# Github Actions

## Publish and Release

When starting a new Python project on GitHub, it's a good practice to follow a version management policy that adheres to semantic versioning, is automated, and makes it easy to maintain and understand the project's release history. One of the widely adopted version management policies for Python projects is using setuptools_scm with semantic versioning and Git tags.

## Set up secrets

1. Store your PyPI API token as a secret in your GitHub repository. Go to your repository's "Settings" tab, click on "Secrets and variables → Actions" in the left menu, and then click the "New repository secret" button. Create a new secret named `PYPI_API_TOKEN` and paste your PyPI API token as the value.
2. Go to your GitHub account settings, go to "Developer settings → Personal access tokens → Tokens(classic)" , and click "Generate new token". The token needs to have the repo scopes but doesn’t need any of the others. Then go to the project’s GitHub Actions secrets and create a new secret named `PERSONAL_ACCESS_TOKEN` with the value of the token you just generated.

## User guid

**Manual Release**

To trigger the workflow `manual_release.yml` manually:

1. Go to the "Actions" tab in your GitHub repository.
2. Select the "Manual Release" workflow from the list of workflows.
3. Click the "Run workflow" button.
4. Choose the branch you want to run the workflow on.
5. Provide the next release version in the input field, or just skip.
6. Click the "Run workflow" button to start the workflow.

GitHub Actions will run the workflow, create a new release with the specified version, and publish it to PyPI. If you skip input it use setuptools_scm to guess next version.

**Auto Release**

The workflow `auto_release.yml` was created to automatically create a new release when a pull request with the label 'release' is merged into the main branch. Auto release also publish package to PyPI.

## Git tags

Use semantic versioning for Git tags:

 - Major version (X): Increment when you make incompatible API changes.
 - Minor version (Y): Increment when you add functionality in a backward-compatible manner.
 - Patch version (Z): Increment when you make backward-compatible bug fixes.

Manual release workflow can input new release version through UI, like `0.3.0`, then create a Git tag `v0.3.0`. When skipping input version, it use setuptools_scm to guess next version. And auto release workflow also use setuptools_scm to guess nex version.

Setuptools_scm only increase patch version. If you want to increase minor version or major version you have to use manual release workflow to input a new version.

## Roadmap

- [ ] Enhance to diplay tag name on title of workflow log when running `Manual Release`.
- [ ] Add test workflow

