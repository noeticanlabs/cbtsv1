import Lake

open Lake DSL

package coherence

lean_lib NoeticanLabs where
  roots := #[`NoeticanLabs]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.27.0"
