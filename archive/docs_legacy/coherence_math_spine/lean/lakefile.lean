import Lake
import Std

package «coherence-math-spine» where
  -- Lean 4 package configuration
  moreServerArgs := #[
    "--memory=2048",
    "--threads=2"
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
  @ "v4.27.0"

@[default_target]
lean_lib «CoherenceMathSpine» where
  -- Lean library configuration
