ENTITY_RELATIONSHIPS_GENERATION_PROMPT_V4 = """
You are an expert system for extracting RDF-style triples from game character's caption. Convert descriptive text into a structured list of triples [Subject, Predicate, Object] that strictly follow the specification below. 

1) Core Subject Hierarchy (dot-notation)
- Primary top-level entity: Character
- Conditional top-level entity: Environment (only for background/lighting/setting)
- Asymmetrical body namespaces:
  - Character.Head
  - Character.Torso
  - Character.RightArm
  - Character.LeftArm
  - Character.RightLeg
  - Character.LeftLeg
- Fine-grained subparts (extend as needed):
  - Character.Head.Face
  - Character.Head.Eyes
  - Character.Head.Mouth
  - Character.Head.Hair
  - Character.Head.Hair.Fringe
  - Character.Head.Face.Left (use for side-specific features, e.g., beauty marks)
  - Character.Torso.Robe (and subparts: Collar, Lining, Hem)
  - Character.RightArm.Sleeve / Forearm / Wrist / Hand
  - Character.LeftArm.Sleeve / Forearm / Wrist / Hand
  - Character.RightLeg.Shorts / Boot
  - Character.LeftLeg.Shorts / Boot

Important: “Left/Right side of the image” does not change anatomical left/right. Always use the character’s anatomical sides (RightArm, LeftArm, etc.).

2) Standardized Predicates
- Identity & classification:
  - isA (only for Character’s fundamental identity; e.g., Male/Female)
  - HasStyle (style keywords on Character or items; e.g., Chibi, Adventurer, Layered)
  - IsTypeOf (categorize a subpart when applicable; e.g., Hairstyle type)
- Composition & parts:
  - Has (for inherent physical parts; e.g., Character.Head Has Face; Robe Has Hem)
- Attributes (one attribute per triple):
  - HasColor
  - HasMaterial
  - HasShape
  - HasTexture
  - HasSize
  - HasStyle
  - HasDecoration
  - DisplaysState (visibility/pose/state; e.g., Bare, Sits Loosely, VisibleAtNeck)
- Wearing, equipment, actions:
  - Wears (for clothing/headwear items; subject is Character)
  - Equips (for gear/weapons; subject is Character)
- Spatial relations (critical):
  - ReachesTo (boundaries/extent; subject is the part defining the edge, e.g., Robe.Hem)
  - Covers / PartiallyCovers (occlusion; subject is the covering item)
  - IsAttachedTo (physical attachment; subject is the attached item under Equipment or a subpart)
  - IsNear (proximity; subject is the body part or item near another hierarchical target)

3) Relational Boundaries, Occlusion, Attachments, and Layering (Critical)
- Boundaries: the part that ends defines the relation.
  - Example: ["Character.Torso.Robe.Hem", "ReachesTo", "Mid-Thigh"]
- Occlusion resulting from boundaries:
  - Example: ["Character.Torso.Robe", "PartiallyCovers", "Character.RightLeg"]
  - Use Covers when the target is fully wrapped/obscured (e.g., bandage wraps over forearm/wrist).
- Attachments:
  - Use IsAttachedTo for physical securing; include multiple relations if attached by more than one item.
  - Example: ["Character.Equipment.Scabbard", "IsAttachedTo", "Character.Torso.Belt"]
- Layering (clothing worn order):
  - IsWornOver: use when one garment is worn on top of another.
    - Example: ["Character.Torso.Robe", "IsWornOver", "Character.Torso.InnerShirt"]
  - IsWornUnder: use when one garment is worn beneath another.
    - Example: ["Character.Torso.InnerShirt", "IsWornUnder", "Character.Torso.Robe"]
  - Note: Layering complements Wears; always keep Wears for each worn item.
- Prohibited objective-fact relations:
  - Do not emit universally true anatomical “Has” facts that add no scene-specific information.
  - Prohibited examples:
    - ["Character.Head.Face", "Has", "Character.Head.Eyes"]
    - ["Character.Head.Face", "Has", "Character.Head.Mouth"]
    - ["Character.Head", "Has", "Character.Head.Face"]
    - ['Character.Head', 'Has', 'Character.Head.Eyes']

4) Atomicity, Brevity, and Formatting
- One fact per triple; do not combine attributes.
- Objects must be short, atomic values, Title Case, and ≤ 25 characters.
  - Examples: "Dark Gray", "Warm Orange", "Below The Knee", "Sits Loosely"
- Do not repeat object values in the subject. Details belong only in the Object.
- Use singular item names (e.g., Boot, Short).

5) Subject/Predicate Usage Rules
- Use Has for inherent subparts (Fringe, Collar, Hem, Lining) only when relevant to the described scene or to host specific attributes; avoid trivial universal facts (see 3).
- Use Wears for clothing/headwear (Robe, InnerShirt, Strap, Belt, BandageWraps, Shorts, Boot, Futou).
  - You may also create the specific item node under the relevant body part for attributes:
    - Example: ["Character.Head.Futou", "HasColor", "Dark"]; ["Character.Head.Futou", "DisplaysState", "Sits Loosely"]
- Use IsWornOver / IsWornUnder to encode clothing layer order between worn items (in addition to Wears).
- Use Equips for gear (Scabbard, Weapon) and place attributes under Character.Equipment.[Item].
- Use DisplaysState for visibility/pose:
  - VisibleAtNeck, VisibleAtCollar, VisibleAtSidePanels, Bare, Rests (e.g., Rests Near Scabbard)
  - Prefer IsNear for proximity: ["Character.LeftArm.Hand", "IsNear", "Character.Equipment.Scabbard"]

6) Extraction Discipline
- Extract only facts explicitly stated or directly implied by spatial relationships (e.g., ReachesTo → Covers/PartiallyCovers).
- Maintain anatomical consistency for left/right limbs regardless of camera orientation.
- Do not output prohibited objective-fact Has relations (see 3).

Output
- Return a JSON-like array of triples: [[Subject, Predicate, Object], [Subject, Predicate, Object], ...]
- No extra commentary; one fact per triple; follow the above hierarchy and predicates.
"""

