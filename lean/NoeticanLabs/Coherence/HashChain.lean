import Mathlib.Data.ByteArray

/-!
# Hash Chain (Axiomatized)

Opaque hash chain API for receipt verification.
Actual crypto implementation deferred to trusted runtime.
-/

namespace NoeticanLabs.Coherence

/-- Hash type (opaque) -/
axiom Hash : Type

/-- Hash function -/
axiom hash : ByteArray → Hash

/-- Chain verification -/
axiom verify_chain : List Hash → Bool

end NoeticanLabs.Coherence
