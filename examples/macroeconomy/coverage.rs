#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageStatus {
    Implemented,
    ValidationOnly,
    ReplicationBlocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EquationCoverageEntry {
    pub equation: u16,
    pub appendix: &'static str,
    pub status: CoverageStatus,
    pub source: &'static str,
    pub note: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EquationCoverage {
    pub entries: Vec<EquationCoverageEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppendixCoverageEntry {
    pub appendix: &'static str,
    pub module: &'static str,
    pub status: CoverageStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicationBlocker {
    pub id: &'static str,
    pub equations: &'static [u16],
    pub source: &'static str,
    pub summary: &'static str,
    pub policy: &'static str,
    pub effect: &'static str,
}

impl EquationCoverage {
    pub fn paper_aligned() -> Self {
        let mut entries = Vec::with_capacity(142);
        for equation in 1..=142 {
            entries.push(EquationCoverageEntry {
                equation,
                appendix: appendix_for_equation(equation),
                status: status_for_equation(equation),
                source: source_for_equation(equation),
                note: note_for_equation(equation),
            });
        }
        Self { entries }
    }

    pub fn missing_equations(&self) -> Vec<u16> {
        let mut seen = [false; 143];
        for entry in &self.entries {
            if (1..=142).contains(&entry.equation) {
                seen[entry.equation as usize] = true;
            }
        }
        (1..=142)
            .filter(|equation| !seen[*equation as usize])
            .collect()
    }

    pub fn unresolved_replication_blockers(&self) -> Vec<EquationCoverageEntry> {
        self.entries
            .iter()
            .copied()
            .filter(|entry| entry.status == CoverageStatus::ReplicationBlocked)
            .collect()
    }

    pub fn blocker_log() -> Vec<ReplicationBlocker> {
        vec![
            ReplicationBlocker {
                id: "ar1-missing-data-policy",
                equations: &[16, 17, 18, 19, 20, 21],
                source: "Paper Appendix A.2; thesis Ch. 6 pp. 156-157",
                summary: "The thesis resolves the main AR(1) form as deterministic lag-one autoregression on log levels through t-1, but not the exact policy for missing, zero, negative, or too-short histories.",
                policy: "Pending user-approved edge-case policy. Fixture code currently uses positive finite observations through t-1 and falls back to the previous positive level when fewer than three observations exist.",
                effect: "Different edge-case policies can change expectation paths for countries or fixtures with sparse, revised, missing, or non-positive historical data.",
            },
            ReplicationBlocker {
                id: "ardl-lag-grid-and-preprocessing",
                equations: &[24],
                source: "Paper Appendix A.3; thesis Ch. 6 pp. 163-164",
                summary: "The thesis gives the ARDL-derived error-correction equation and AIC lag selection, but not the lag candidate grid, transformation details, residual handling, or missing-data policy.",
                policy: "Expose the thesis error-correction form, use AIC candidate selection hooks, and require config or author code for exact empirical lag/preprocessing choices.",
                effect: "Different lag grids or preprocessing can change bank loan rates, credit rationing, debt service, defaults, and downstream production and housing outcomes.",
            },
            ReplicationBlocker {
                id: "author-initialization-tie-breaks",
                equations: &[22, 23, 50, 51, 55, 56, 101, 102, 128],
                source: "Paper Section 3 and Appendix A; thesis Ch. 6 pp. 147, 160-176, 192-206",
                summary: "The thesis clarifies data sources, sampling, rescaling, and assignment problems, but not author tie-breaking and exact weighted-sampling conventions.",
                policy: "Fixture mode uses deterministic synthetic data; real-data mode fails fast with named missing assets and requires author-equivalent initialization scripts.",
                effect: "Different tie-breaks or weighted samples can change the initial microstate while preserving the same aggregate totals.",
            },
            ReplicationBlocker {
                id: "credit-visit-limits",
                equations: &[],
                source: "Paper Appendix A.12; thesis Ch. 6 pp. 212-213",
                summary: "The thesis confirms the random subset and ascending-rate visit rule, but does not provide numeric nLF and nLH values.",
                policy: "Expose firm and household bank visit limits in ReplicationPolicy/config; fixture mode defaults both visit limits to 2.",
                effect: "Different visit limits alter the probability of credit approval and the distribution of loans across banks.",
            },
        ]
    }

    pub fn gap_report_text() -> String {
        let blockers = Self::blocker_log();
        let mut text = String::new();
        text.push_str("remaining_exact_replication_gaps\n");
        if blockers.is_empty() {
            text.push_str("none\n");
            return text;
        }
        for blocker in blockers {
            let equations = format_equations(blocker.equations);
            text.push_str(&format!(
                "- gap_id: {}\n  source: {}\n  affected: {}\n  current_policy: {}\n  effect: {}\n",
                blocker.id, blocker.source, equations, blocker.policy, blocker.effect
            ));
        }
        text
    }

    pub fn gap_report_json() -> String {
        let mut text = String::from("[");
        for (idx, blocker) in Self::blocker_log().iter().enumerate() {
            if idx > 0 {
                text.push(',');
            }
            text.push_str(&format!(
                "{{\"gap_id\":\"{}\",\"source\":\"{}\",\"affected\":\"{}\",\"current_policy\":\"{}\",\"effect\":\"{}\"}}",
                json_escape(blocker.id),
                json_escape(blocker.source),
                json_escape(&format_equations(blocker.equations)),
                json_escape(blocker.policy),
                json_escape(blocker.effect)
            ));
        }
        text.push(']');
        text
    }

    pub fn appendix_labels() -> [&'static str; 13] {
        [
            "A.1 aggregates",
            "A.2 expectations",
            "A.3 banks",
            "A.4 central bank",
            "A.5 firms",
            "A.6 government",
            "A.7 households/properties",
            "A.8 individuals",
            "A.9 rest of world",
            "A.10 goods market",
            "A.11 labour market",
            "A.12 credit market",
            "A.13 housing market",
        ]
    }

    pub fn appendix_coverage() -> Vec<AppendixCoverageEntry> {
        vec![
            AppendixCoverageEntry {
                appendix: "A.1 aggregates",
                module: "aggregate_previous_state / realised_accounting",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.2 expectations",
                module: "refit_expectations",
                status: CoverageStatus::ReplicationBlocked,
            },
            AppendixCoverageEntry {
                appendix: "A.3 banks",
                module: "credit_market / realised_accounting",
                status: CoverageStatus::ReplicationBlocked,
            },
            AppendixCoverageEntry {
                appendix: "A.4 central bank",
                module: "planning_and_production",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.5 firms",
                module: "target_setting / planning_and_production / realised_accounting",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.6 government",
                module: "planning_and_production / realised_accounting",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.7 households/properties",
                module: "planning_and_production / housing_preclear / housing_completion",
                status: CoverageStatus::ReplicationBlocked,
            },
            AppendixCoverageEntry {
                appendix: "A.8 individuals",
                module: "target_setting / planning_and_production",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.9 rest of world",
                module: "planning_and_production / goods_market",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.10 goods market",
                module: "goods_market",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.11 labour market",
                module: "labour_market",
                status: CoverageStatus::Implemented,
            },
            AppendixCoverageEntry {
                appendix: "A.12 credit market",
                module: "credit_market",
                status: CoverageStatus::ReplicationBlocked,
            },
            AppendixCoverageEntry {
                appendix: "A.13 housing market",
                module: "housing_preclear / housing_completion",
                status: CoverageStatus::Implemented,
            },
        ]
    }
}

impl Default for EquationCoverage {
    fn default() -> Self {
        Self::paper_aligned()
    }
}

fn appendix_for_equation(equation: u16) -> &'static str {
    match equation {
        1..=15 => "A.1 aggregates",
        16..=21 => "A.2 expectations",
        22..=44 => "A.3 banks",
        45..=49 => "A.4 central bank",
        50..=94 => "A.5 firms",
        95..=100 => "A.6 government",
        101..=127 => "A.7 households/properties",
        128..=133 => "A.8 individuals",
        134..=139 => "A.9 rest of world",
        140 => "A.10 goods market",
        141..=142 => "A.11 labour market",
        _ => "unknown",
    }
}

fn status_for_equation(equation: u16) -> CoverageStatus {
    match equation {
        15 => CoverageStatus::ValidationOnly,
        24 => CoverageStatus::ReplicationBlocked,
        _ => CoverageStatus::Implemented,
    }
}

fn source_for_equation(equation: u16) -> &'static str {
    match equation {
        16..=21 => "Paper Appendix A.2; thesis Ch. 6 pp. 156-157",
        22..=44 => "Paper Appendix A.3; thesis Ch. 6 pp. 159-168",
        45..=49 => "Paper Appendix A.4; thesis Ch. 6 pp. 168-169",
        50..=94 => "Paper Appendix A.5; thesis Ch. 6 pp. 170-185",
        95..=100 => "Paper Appendix A.6; thesis Ch. 6 pp. 185-188",
        101..=127 => "Paper Appendix A.7; thesis Ch. 6 pp. 188-204",
        128..=133 => "Paper Appendix A.8; thesis Ch. 6 pp. 204-207",
        134..=139 => "Paper Appendix A.9; thesis Ch. 6 pp. 208-209",
        140 => {
            "Paper Appendix A.10; thesis Ch. 6 pp. 210-211; Poledna et al. Online Appendix A.1.1"
        }
        141..=142 => "Paper Appendix A.11; thesis Ch. 6 pp. 211-212",
        _ => "Paper Appendix A; thesis Ch. 6",
    }
}

fn note_for_equation(equation: u16) -> &'static str {
    match equation {
        15 => "GDP identity is checked as an invariant/report instead of written as state.",
        16..=21 => "Thesis-aligned deterministic AR(1) on log levels through t-1 is implemented; sparse-history edge policy is reported as a gap.",
        24 => "Thesis ARDL error-correction form is represented; exact preprocessing, lag candidates, and residual policy are gap-reported.",
        42 | 43 => "Bank liability and reserve equations are encoded directly with explicit positive/negative-part helpers.",
        109 => "Annual purchase-cost formula is encoded literally from the visually checked PDF/OCR snippet.",
        113 | 115 => "Literal housing price/rent reduction equations are available as the default policy.",
        140 => "Implemented with the Poledna inherited goods search rule: random consumer ordering, seller probability averaged from exp(-phi_GM * price) and relative firm size, fallback to remaining sellers, and excess-demand reporting.",
        141 | 142 => "Random market ordering uses a Python-like MT19937 stream; tied ranked choices are randomized by seeded shuffle/Bernoulli policy where explicit ties remain.",
        _ => "Paper-specified behavior implemented in the example systems or fixture initialisation.",
    }
}

fn format_equations(equations: &[u16]) -> String {
    if equations.is_empty() {
        return "non-equation procedural rule".to_owned();
    }
    equations
        .iter()
        .map(|equation| format!("A.{equation}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            _ => vec![ch],
        })
        .collect()
}
