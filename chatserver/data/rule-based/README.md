# Rule-based Auto Data Cleaning

We score or classify text quality with a prompt which is based on certain rules.
Then we use third-party APIs to execute this rule.

By doing this, we are NOT training our model by directly extracting data from third-party APIs by ourselves, which may violate terms of use.
Instead, we simply filter out harmful content, which is crucial for AI safety.
