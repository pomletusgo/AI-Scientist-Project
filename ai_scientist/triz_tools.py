#!/usr/bin/env python3
"""
TRIZ Tool-Augmented Functions for AI-Scientist.
LLM calls these tools step-by-step instead of reading a 40KB knowledge dump.
Each function returns structured results that the LLM uses to build ideas.
"""

import os, json

# ============================================================
# Data: 39 Engineering Parameters
# ============================================================
PARAMETERS = {
    1:  "Weight of moving object",
    2:  "Weight of stationary object",
    3:  "Length of moving object",
    4:  "Length of stationary object",
    5:  "Area of moving object",
    6:  "Area of stationary object",
    7:  "Volume of moving object",
    8:  "Volume of stationary object",
    9:  "Speed",
    10: "Force",
    11: "Stress or pressure",
    12: "Shape",
    13: "Stability of the object's composition",
    14: "Strength",
    15: "Duration of action by a moving object",
    16: "Duration of action by a stationary object",
    17: "Temperature",
    18: "Illumination intensity",
    19: "Use of energy by moving object",
    20: "Use of energy by stationary object",
    21: "Power",
    22: "Loss of Energy",
    23: "Loss of substance",
    24: "Loss of Information",
    25: "Loss of Time",
    26: "Quantity of substance/the matter",
    27: "Reliability",
    28: "Measurement accuracy",
    29: "Manufacturing precision",
    30: "External harm affects the object",
    31: "Object-generated harmful factors",
    32: "Ease of manufacture",
    33: "Ease of operation",
    34: "Ease of repair",
    35: "Adaptability or versatility",
    36: "Device complexity",
    37: "Difficulty of detecting and measuring",
    38: "Extent of automation",
    39: "Productivity",
}

# ML-domain parameter mapping for auto-suggestion
ML_PARAMETER_MAP = {
    "accuracy": 28, "precision": 28, "recall": 39, "f1": 28,
    "speed": 9, "latency": 25, "training_time": 25, "inference_time": 25,
    "model_size": 26, "parameters": 26, "memory": 26,
    "robustness": 27, "reliability": 27, "stability": 13,
    "generalization": 35, "adaptability": 35,
    "complexity": 36, "simplicity": 32,
    "energy": 19, "compute": 19, "cost": 22,
    "overfitting": 13, "bias": 30,
    "fairness": 30, "interpretability": 37, "explainability": 37,
}

# ============================================================
# Data: 40 Inventive Principles (abbreviated for tool response)
# ============================================================
PRINCIPLES = {
    1:  ("Segmentation",
         "Divide into independent parts; make easy to disassemble; increase fragmentation.",
         "Mixture-of-Experts, model sharding, modular networks, independent feature extractors."),
    2:  ("Extraction / Taking Out",
         "Extract the harmful or necessary part; remove the most important part to create something new.",
         "Knowledge distillation, pruning, feature selection, adapter modules, low-rank factorization."),
    3:  ("Local Quality",
         "Change uniform to non-uniform; let different parts perform different functions.",
         "Attention mechanisms, per-layer learning rates, heterogeneous GNNs, localized loss."),
    4:  ("Asymmetry",
         "Change symmetrical to asymmetrical; increase degree of asymmetry.",
         "Asymmetric encoder-decoder, focal loss (asymmetric penalty), class-weighted sampling."),
    5:  ("Merging / Combining",
         "Combine identical objects in space/time; combine heterogeneous objects.",
         "Multi-task learning, ensembles, hybrid architectures (CNN+Transformer), multi-modal models."),
    6:  ("Universality / Multi-functionality",
         "One object performs multiple functions, eliminating others.",
         "Foundation models, multi-head architectures, general-purpose agents."),
    7:  ("Nesting / Matryoshka",
         "Place one object inside another; one part retracts into another.",
         "Residual connections, hierarchical representations, recursive networks, nested attention."),
    8:  ("Anti-weight / Weight Compensation",
         "Merge with lifting force; balance via medium.",
         "Gradient clipping, batch normalization, residual connections, learning rate warmup."),
    9:  ("Preliminary Anti-action",
         "Apply anti-action in advance; pre-apply compression if tension is expected.",
         "Adversarial training, gradient pre-conditioning, data augmentation as pre-normalization."),
    10: ("Preliminary Action",
         "Perform changes in advance; pre-arrange for timely use.",
         "Pre-training, transfer learning, pre-computed embeddings, caching, model warm-up."),
    11: ("Beforehand Cushioning",
         "Prepare emergency measures for low-reliability objects.",
         "Checkpointing, early stopping, fallback models, redundancy, graceful degradation."),
    12: ("Equipotentiality",
         "Eliminate need to raise/lower objects; change working height.",
         "Zero-centering, batch norm, balanced sampling, learning rate equalization."),
    13: ("Inversion / The Other Way Round",
         "Use opposite action; make moving parts fixed; turn inside-out.",
         "GANs, inverse RL, self-supervised learning, generate-then-verify."),
    14: ("Spheroidality / Curvature",
         "Use curves instead of lines; balls instead of flats; rotary instead of linear.",
         "Kernel methods (RBF), circular convolutions, geodesic distance in manifolds."),
    15: ("Dynamicity",
         "Allow adjustment for optimal performance; make rigid objects adaptive.",
         "Adaptive learning rates, dynamic architectures, learnable augmentation, online learning."),
    16: ("Partial or Excessive Action",
         "Use slightly less or more than 100% to simplify.",
         "Over-parameterize then prune, curriculum learning, warm-start from simplified problem."),
    17: ("Another Dimension / Dimensionality Change",
         "Move to 2D/3D; multi-layer instead of single; tilt or re-orient.",
         "Multi-scale representations, 2D conv for images, multi-head attention, positional encoding."),
    18: ("Mechanical Vibration",
         "Oscillate; increase frequency to ultrasonic; use resonance.",
         "Cyclical learning rates, cosine annealing with restarts, SGD noise as beneficial vibration."),
    19: ("Periodic Action",
         "Use pulses instead of continuous; change frequency; use pauses for other actions.",
         "Cosine annealing schedules, periodic evaluation, alternating GAN training."),
    20: ("Continuity of Useful Action",
         "All parts work continuously; eliminate idle time.",
         "Pipeline parallelism, streaming inference, continuous learning, async training."),
    21: ("Skipping / Rushing Through",
         "Conduct harmful operations at high speed.",
         "Fast initial training with large LR, one-cycle policy, rapid prototyping."),
    22: ("Blessing in Disguise / Turn Harm into Benefit",
         "Use harmful factors positively; eliminate one harm by adding another.",
         "Adversarial examples for robust training, noise injection for regularization."),
    23: ("Feedback",
         "Introduce feedback; change its magnitude.",
         "Validation-based early stopping, LR reduction on plateau, RL, active learning."),
    24: ("Intermediary",
         "Use intermediary carrier; temporarily merge with removable object.",
         "Multi-agent systems, adapter modules, proxy models, bottleneck layers."),
    25: ("Self-service",
         "Object serves itself; use waste resources.",
         "Self-supervised learning, AutoML, self-healing, bootstrapping, pseudo-labeling."),
    26: ("Copying",
         "Use simpler copies instead of expensive originals; optical/IR copies.",
         "Knowledge distillation, synthetic data, model compression, quantization, digital twins."),
    27: ("Cheap Short-lived Objects",
         "Replace expensive with many inexpensive short-lived ones.",
         "Lightweight edge models, disposable hyperparameter runs, snapshot ensembles."),
    28: ("Substitution for Mechanical Means",
         "Replace mechanical with sensory/electrical/magnetic/fields.",
         "Replace rule-based with learned functions, replace manual features with representation learning."),
    29: ("Pneumatics and Hydraulics",
         "Use gas/liquid instead of solid.",
         "Soft attention, Gumbel-softmax, continuous relaxation of discrete operations."),
    30: ("Flexible Shells and Thin Films",
         "Replace heavy structures with flexible shells; isolate with films.",
         "Adapter layers, LoRA, prefix tuning, parameter-efficient fine-tuning."),
    31: ("Porous Materials",
         "Make porous; fill pores with useful substance.",
         "Sparse networks, dropout, sparse attention, model sparsity, grafting."),
    32: ("Changing Color / Optical Properties",
         "Change color or transparency.",
         "Model interpretability, attention visualization, saliency maps, t-SNE coloring."),
    33: ("Homogeneity",
         "Use same material for interacting objects.",
         "Siamese networks, shared weights, contrastive learning, homogeneous GNNs."),
    34: ("Discarding and Recovering",
         "Eliminate parts after function; restore consumables during operation.",
         "Pruning, dynamic layer removal, early-exit networks, weight recycling."),
    35: ("Parameter Change",
         "Change aggregate state, concentration, flexibility, temperature.",
         "Temperature scaling, LR schedules, precision change (FP32→FP16), annealing."),
    36: ("Phase Transitions",
         "Use phenomena during phase changes (volume, heat).",
         "Grokking, critical learning periods, emergence in large models."),
    37: ("Thermal Expansion",
         "Use expansion/contraction; multiple materials with different coefficients.",
         "Progressive network growing, embedding expansion, staged training."),
    38: ("Strong Oxidizers",
         "Enrich, ionize, ozonize.",
         "Enriched/higher-quality training data, amplified gradients for rare classes."),
    39: ("Inert Atmosphere",
         "Use inert gas; add neutral parts.",
         "Controlled baselines, ablation studies, noise for stability."),
    40: ("Composite Materials",
         "Use composites instead of homogeneous materials.",
         "Hybrid architectures, ensemble diversity, multi-branch networks, MoE with different expert types."),
}

# ============================================================
# Contradiction Matrix (abbreviated: key cells for ML)
# ============================================================
# Format: (improving_param, worsening_param) → [principle_numbers]
# Based on Altshuller's official 39×39 matrix
MATRIX = {
    (9, 25):  [28, 30, 36, 2],      # Speed vs Loss of Time
    (9, 27):  [11, 35, 27, 28],     # Speed vs Reliability
    (14, 2):  [28, 29, 35, 26],     # Strength vs Weight(stationary)
    (14, 26): [3, 18, 27, 40],      # Strength vs Quantity of substance
    (15, 25): [20, 10, 28, 18],     # Duration vs Loss of Time
    (25, 9):  [28, 30, 36, 2],      # Loss of Time vs Speed
    (25, 26): [4, 10, 27, 22],      # Loss of Time vs Quantity
    (26, 25): [10, 35, 20, 28],     # Quantity vs Loss of Time
    (27, 25): [10, 30, 4],          # Reliability vs Loss of Time
    (27, 26): [3, 10, 8, 28],       # Reliability vs Quantity
    (27, 36): [13, 35, 1],          # Reliability vs Device complexity
    (28, 25): [24, 34, 28, 32],     # Measurement accuracy vs Loss of Time
    (28, 36): [27, 9, 26, 24],      # Measurement accuracy vs Complexity
    (32, 25): [35, 28, 34, 4],      # Ease of manufacture vs Loss of Time
    (33, 25): [4, 28, 10, 34],      # Ease of operation vs Loss of Time
    (35, 25): [35, 28],             # Adaptability vs Loss of Time
    (35, 36): [15, 28, 37, 1],      # Adaptability vs Complexity
    (36, 25): [28, 10, 35, 23],     # Complexity vs Loss of Time
    (36, 27): [13, 35, 1],          # Complexity vs Reliability
    (38, 25): [10, 28, 35],         # Automation vs Loss of Time
    (38, 36): [15, 24, 10],         # Automation vs Complexity
    (39, 25): [10, 28, 35, 23],     # Productivity vs Loss of Time
    (39, 27): [1, 35, 29, 28],      # Productivity vs Reliability
    (39, 36): [12, 17, 28, 24],     # Productivity vs Complexity
}

# ============================================================
# Tool 1: Look up contradiction matrix
# ============================================================
def lookup_contradiction_matrix(improving_param, worsening_param):
    """Return recommended principles for a contradiction."""
    key = (improving_param, worsening_param)
    if key in MATRIX:
        principles = MATRIX[key]
    else:
        # Fallback: search by principle frequency
        principles = [1, 35, 15]  # Common defaults for unknown pairs

    result = {
        "contradiction": {
            "improving": {"id": improving_param,
                          "name": PARAMETERS.get(improving_param, f"Parameter {improving_param}")},
            "worsening": {"id": worsening_param,
                          "name": PARAMETERS.get(worsening_param, f"Parameter {worsening_param}")},
        },
        "recommended_principles": [
            {"number": p, "name": PRINCIPLES[p][0]}
            for p in principles
        ],
    }
    return result


# ============================================================
# Tool 2: Get full principle detail
# ============================================================
def get_principle_detail(principle_number):
    """Return complete description + ML applications for a principle."""
    if principle_number not in PRINCIPLES:
        return {"error": f"Principle {principle_number} not found. Valid range: 1-40."}

    name, description, ml_applications = PRINCIPLES[principle_number]
    return {
        "number": principle_number,
        "name": name,
        "description": description,
        "ml_applications": ml_applications,
        "how_to_apply": (
            f"Principle #{principle_number} ({name}): {description} "
            f"In ML, this maps to: {ml_applications} "
            f"To apply, ask: how can I {description.lower().rstrip('.')} "
            f"in the context of the given model architecture and training pipeline?"
        ),
    }


# ============================================================
# Tool 3: Suggest ML parameters for a given concept
# ============================================================
def map_ml_concept_to_parameters(concept):
    """Map a common ML concept to the closest TRIZ 39 parameters."""
    concept_lower = concept.lower()
    matches = []
    for keyword, param_id in ML_PARAMETER_MAP.items():
        if keyword in concept_lower or concept_lower in keyword:
            matches.append({
                "keyword": keyword,
                "parameter_id": param_id,
                "parameter_name": PARAMETERS[param_id],
            })

    if not matches:
        return {
            "concept": concept,
            "suggestion": "Try a more specific term: accuracy, speed, robustness, complexity, "
                          "generalization, interpretability, model_size, latency, overfitting, "
                          "fairness, compute, energy",
        }

    return {"concept": concept, "matches": matches[:5]}


# ============================================================
# Tool 4: Evaluate ideality score
# ============================================================
def evaluate_ideality(benefits, costs, harms):
    """
    Calculate Ideality = Benefits / (Costs + Harm).
    Each input is a list of {name: str, score: float} where score is 1-10.
    """
    b_sum = sum(item.get("score", 0) for item in benefits)
    c_sum = sum(item.get("score", 0) for item in costs)
    h_sum = sum(item.get("score", 0) for item in harms)

    denominator = c_sum + h_sum
    if denominator == 0:
        denominator = 1

    ideality = round(b_sum / denominator, 3)

    if ideality >= 2.0:
        level = "Excellent — strongly recommend pursuing"
    elif ideality >= 1.0:
        level = "Good — worth investigating"
    elif ideality >= 0.5:
        level = "Marginal — needs refinement to reduce costs/harm"
    else:
        level = "Poor — likely not worth the trade-offs"

    return {
        "benefits_total": b_sum,
        "costs_total": c_sum,
        "harms_total": h_sum,
        "ideality_score": ideality,
        "level": level,
        "formula": f"Ideality = {b_sum} / ({c_sum} + {h_sum}) = {ideality}",
    }


# ============================================================
# Tool 5: Check principle novelty in domain
# ============================================================
def check_principle_novelty(principle_number, domain):
    """
    Estimate how commonly a principle is used in a given ML domain.
    Returns a novelty level for the idea.
    """
    # Heuristic: principles commonly used in ML are less novel in any domain
    COMMON_ML_PRINCIPLES = {1, 2, 5, 7, 10, 13, 15, 23, 25, 26, 28, 35, 40}
    RARE_ML_PRINCIPLES = {4, 8, 9, 11, 12, 14, 16, 18, 19, 21, 22, 27, 29,
                          30, 31, 32, 33, 34, 36, 37, 38, 39}

    name = PRINCIPLES.get(principle_number, ("Unknown",))[0]

    if principle_number in RARE_ML_PRINCIPLES:
        novelty = "HIGH — this principle is rarely applied in ML, giving your idea strong novelty"
    elif principle_number in COMMON_ML_PRINCIPLES:
        novelty = "MODERATE — commonly used in ML; combine with a rare principle for higher novelty"
    else:
        novelty = "MODERATE"

    return {
        "principle": f"#{principle_number} {name}",
        "domain": domain,
        "novelty_assessment": novelty,
    }


# ============================================================
# Tool 6: Find contradictions for a domain
# ============================================================
def suggest_contradictions(domain, task_description=""):
    """
    Suggest the 3 most likely technical contradictions for a given domain.
    Supports ML domains + cross-discipline (physics, biology, chemistry,
    medicine, economics, materials, engineering).
    Returns structured contradictions ready for matrix lookup.
    """
    # ================================================================
    # Domain-specific presets — ML + cross-discipline
    # ================================================================
    DOMAIN_PRESETS = {
        # --- ML domains ---
        "churn": [
            {"improving": 39, "worsening": 27, "statement": "Increasing recall (catching more churners) reduces precision (more false alarms)"},
            {"improving": 28, "worsening": 25, "statement": "Improving prediction accuracy requires more training/inference time"},
            {"improving": 35, "worsening": 36, "statement": "Making the model more adaptable to different customer segments increases complexity"},
        ],
        "classification": [
            {"improving": 39, "worsening": 27, "statement": "Increasing classification recall reduces precision"},
            {"improving": 28, "worsening": 25, "statement": "Improving accuracy increases training time"},
            {"improving": 35, "worsening": 36, "statement": "Better generalization requires more complex models"},
        ],
        "detection": [
            {"improving": 28, "worsening": 25, "statement": "More accurate detection takes more time"},
            {"improving": 39, "worsening": 27, "statement": "Higher detection rate produces more false positives"},
            {"improving": 35, "worsening": 26, "statement": "Adapting to diverse inputs requires more parameters"},
        ],
        "regression": [
            {"improving": 28, "worsening": 25, "statement": "More precise predictions require more computation"},
            {"improving": 35, "worsening": 36, "statement": "Handling non-linear patterns increases model complexity"},
            {"improving": 39, "worsening": 22, "statement": "More predictions per second consumes more energy"},
        ],
        "nlp": [
            {"improving": 28, "worsening": 26, "statement": "Better language understanding requires larger models"},
            {"improving": 39, "worsening": 25, "statement": "Faster text processing sacrifices accuracy"},
            {"improving": 35, "worsening": 36, "statement": "Multi-lingual support increases architectural complexity"},
        ],
        "vision": [
            {"improving": 28, "worsening": 25, "statement": "Higher resolution processing takes more time"},
            {"improving": 39, "worsening": 26, "statement": "More images per batch requires more GPU memory"},
            {"improving": 35, "worsening": 36, "statement": "Robustness to lighting/angles complicates the architecture"},
        ],
        # --- Cross-discipline domains ---
        # Physics: optics / imaging
        "physics": [
            {"improving": 28, "worsening": 25, "statement": "Higher measurement resolution requires longer integration time"},
            {"improving": 14, "worsening": 2,   "statement": "Stronger materials (higher tensile strength) are typically heavier (higher density)"},
            {"improving": 17, "worsening": 22, "statement": "Higher operating temperature increases energy loss through thermal radiation"},
        ],
        "optics": [
            {"improving": 28, "worsening": 25, "statement": "Higher image resolution requires longer exposure or more sensitive detectors"},
            {"improving": 18, "worsening": 22, "statement": "Brighter illumination increases energy consumption and heat generation"},
            {"improving": 28, "worsening": 36, "statement": "Better aberration correction makes optical systems more complex"},
        ],
        # Biology: genomics / sequencing
        "biology": [
            {"improving": 28, "worsening": 25, "statement": "Higher sequencing depth increases accuracy but requires more time and resources"},
            {"improving": 39, "worsening": 27, "statement": "Higher throughput screening produces more false positives"},
            {"improving": 35, "worsening": 36, "statement": "Broader pathogen detection panel increases assay complexity"},
        ],
        "genomics": [
            {"improving": 28, "worsening": 22, "statement": "Higher sequencing accuracy requires more reagent consumption (higher cost)"},
            {"improving": 39, "worsening": 27, "statement": "Faster variant calling increases false discovery rate"},
            {"improving": 26, "worsening": 25, "statement": "Larger reference databases improve annotation but slow down alignment"},
        ],
        # Chemistry: catalysis / synthesis
        "chemistry": [
            {"improving": 39, "worsening": 27, "statement": "Higher reaction rate reduces selectivity towards desired product"},
            {"improving": 17, "worsening": 13, "statement": "Higher reaction temperature degrades catalyst stability over time"},
            {"improving": 32, "worsening": 22, "statement": "Simpler synthesis routes often use more expensive precursors"},
        ],
        "catalysis": [
            {"improving": 15, "worsening": 13, "statement": "Longer catalyst lifetime requires more robust (and expensive) materials"},
            {"improving": 39, "worsening": 28, "statement": "Higher throughput reduces reaction control precision"},
            {"improving": 14, "worsening": 17, "statement": "Stronger catalyst binding increases operating temperature requirements"},
        ],
        # Medicine / diagnostics
        "medicine": [
            {"improving": 28, "worsening": 30, "statement": "More accurate diagnostic tests are often more invasive for the patient"},
            {"improving": 39, "worsening": 27, "statement": "Faster screening throughput increases false positive rate"},
            {"improving": 33, "worsening": 36, "statement": "Easier-to-use point-of-care devices have lower multiplexing capability"},
        ],
        "diagnostics": [
            {"improving": 28, "worsening": 25, "statement": "Higher diagnostic sensitivity requires longer assay time"},
            {"improving": 32, "worsening": 22, "statement": "Cheaper manufacturing processes increase per-test reagent cost"},
            {"improving": 27, "worsening": 36, "statement": "More reliable multi-marker panels require more complex detection systems"},
        ],
        # Economics / social science
        "economics": [
            {"improving": 28, "worsening": 36, "statement": "More accurate economic models require exponentially more variables and complexity"},
            {"improving": 35, "worsening": 27, "statement": "Models that generalize across markets lose predictive reliability in specific cases"},
            {"improving": 37, "worsening": 25, "statement": "Better causal identification requires longer data collection periods"},
        ],
        # Materials science
        "materials": [
            {"improving": 14, "worsening": 1,  "statement": "Stronger materials are typically heavier (higher weight)"},
            {"improving": 17, "worsening": 13, "statement": "Higher thermal resistance reduces material stability under thermal cycling"},
            {"improving": 35, "worsening": 32, "statement": "Multi-functional materials are harder to manufacture at scale"},
        ],
        # Engineering / systems
        "engineering": [
            {"improving": 27, "worsening": 36, "statement": "More reliable systems require redundant components, increasing complexity"},
            {"improving": 21, "worsening": 22, "statement": "Higher power output increases energy losses through heat dissipation"},
            {"improving": 33, "worsening": 32, "statement": "Easier operation typically requires more sophisticated manufacturing"},
        ],
        # Environmental science
        "environmental": [
            {"improving": 28, "worsening": 25, "statement": "More accurate climate models require longer simulation runs"},
            {"improving": 39, "worsening": 27, "statement": "Faster remediation of pollutants increases risk of secondary contamination"},
            {"improving": 19, "worsening": 30, "statement": "Higher energy efficiency of renewable systems increases land use impact"},
        ],
    }

    key = domain.lower()
    best = "classification"  # ultimate fallback
    # Try exact match first, then substring match
    for preset in DOMAIN_PRESETS:
        if preset == key or preset in key or key in preset:
            best = preset
            break

    presets = DOMAIN_PRESETS[best]
    result = {"domain": domain, "matched_preset": best, "contradictions": []}
    for c in presets:
        entry = dict(c)
        mp = lookup_contradiction_matrix(c["improving"], c["worsening"])
        entry["recommended_principles"] = mp["recommended_principles"]
        result["contradictions"].append(entry)

    return result


# ============================================================
# Tool definitions for OpenAI function-calling format
# ============================================================
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "suggest_contradictions",
            "description": "For a given ML domain/task, suggest the 3 most likely technical contradictions following TRIZ methodology. Call this FIRST when starting to analyze a new research topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "ML domain name (e.g. churn, classification, detection, nlp, vision)"},
                    "task_description": {"type": "string", "description": "Brief description of the research problem"},
                },
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_contradiction_matrix",
            "description": "Query the TRIZ 39×39 contradiction matrix. Given an improving parameter and a worsening parameter, returns the top 2-4 Inventive Principles that historically resolve this type of contradiction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "improving_param": {"type": "integer", "description": "Parameter ID you want to improve (1-39). Common ML: 9=speed, 25=loss_of_time, 27=reliability, 28=accuracy, 35=adaptability, 36=complexity, 39=productivity"},
                    "worsening_param": {"type": "integer", "description": "Parameter ID that gets worse (1-39)"},
                },
                "required": ["improving_param", "worsening_param"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_principle_detail",
            "description": "Get the complete description, sub-principles, and ML-specific application examples for a given TRIZ Inventive Principle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principle_number": {"type": "integer", "description": "Principle number (1-40)"},
                },
                "required": ["principle_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_principle_novelty",
            "description": "Check how novel it would be to apply a given TRIZ principle in a specific ML domain. High novelty = rarely used = stronger paper contribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principle_number": {"type": "integer", "description": "Principle number (1-40)"},
                    "domain": {"type": "string", "description": "ML domain name"},
                },
                "required": ["principle_number", "domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_ideality",
            "description": "Calculate the Ideality score of a proposed solution: Ideality = ΣBenefits / (ΣCosts + ΣHarm). Each item should have a score from 1-10.",
            "parameters": {
                "type": "object",
                "properties": {
                    "benefits": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "score": {"type": "number"}}}},
                    "costs": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "score": {"type": "number"}}}},
                    "harms": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "score": {"type": "number"}}}},
                },
                "required": ["benefits", "costs", "harms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_ml_concept_to_parameters",
            "description": "Map a common ML concept (e.g. 'accuracy', 'overfitting') to the corresponding TRIZ 39 engineering parameter IDs for use with the contradiction matrix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "ML concept to map (e.g. accuracy, recall, overfitting, model_size, latency)"},
                },
                "required": ["concept"],
            },
        },
    },
]

# Tool execution map: function name → callable
TOOL_MAP = {
    "suggest_contradictions": suggest_contradictions,
    "lookup_contradiction_matrix": lookup_contradiction_matrix,
    "get_principle_detail": get_principle_detail,
    "check_principle_novelty": check_principle_novelty,
    "evaluate_ideality": evaluate_ideality,
    "map_ml_concept_to_parameters": map_ml_concept_to_parameters,
}
