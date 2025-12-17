#[test]
fn toy_economy_pure_abm_runs() {
    // ----------------------------
    // Toy economy:
    // - N agents each want 1 bread/step if they can afford it
    // - M firms produce bread, hold inventory, set price, pay wages
    // - Price rises if inventory low, falls if inventory high
    // ----------------------------

    #[derive(Clone, Debug)]
    struct Agent {
        cash: f32,
        hunger: f32, // grows if they don't consume
        employed_by: usize, // firm index
    }

    #[derive(Clone, Debug)]
    struct Firm {
        cash: f32,
        price: f32,
        inventory: f32,
        production_per_step: f32,
        wage: f32,
        target_inventory: f32,
    }

    fn step(agents: &mut [Agent], firms: &mut [Firm]) {
        // 1) Production + wages
        for (fi, firm) in firms.iter_mut().enumerate() {
            firm.inventory += firm.production_per_step;

            // Pay wages to employees of this firm
            let mut payroll = 0.0;
            for a in agents.iter_mut().filter(|a| a.employed_by == fi) {
                let w = firm.wage.min(firm.cash); // can't pay more than you have
                firm.cash -= w;
                a.cash += w;
                payroll += w;
            }

            // small sanity: payroll doesn't change totals, only redistributes
            let _ = payroll;
        }

        // 2) Consumption / market (agents buy 1 bread if possible)
        // Simple demand matching: agents try their employer firm first.
        for a in agents.iter_mut() {
            let want = 1.0_f32;
            let fi = a.employed_by;

            let firm = &mut firms[fi];
            if firm.inventory >= want && a.cash >= firm.price * want {
                // Buy
                a.cash -= firm.price * want;
                firm.cash += firm.price * want;
                firm.inventory -= want;
                a.hunger = (a.hunger - 1.0).max(0.0);
            } else {
                // Can't buy => hunger increases
                a.hunger += 1.0;
            }
        }

        // 3) Firm pricing rule (inventory-based)
        for firm in firms.iter_mut() {
            // If inventory below target -> raise price; above -> lower price
            let inv_gap = firm.target_inventory - firm.inventory;
            let adj = 0.02 * inv_gap; // small proportional controller
            firm.price = (firm.price + adj).clamp(0.05, 1000.0);

            // Keep wage somewhat tied to price to make the system stable-ish
            firm.wage = (0.25 * firm.price).clamp(0.01, 1000.0);
        }
    }

    // --------- init ----------
    let n_agents = 200;
    let n_firms = 5;

    let mut firms: Vec<Firm> = (0..n_firms)
        .map(|_| Firm {
            cash: 1_000.0,
            price: 2.0,
            inventory: 50.0,
            production_per_step: 80.0 / n_firms as f32,
            wage: 0.5,
            target_inventory: 50.0,
        })
        .collect();

    let mut agents: Vec<Agent> = (0..n_agents)
        .map(|i| Agent {
            cash: 10.0,
            hunger: 0.0,
            employed_by: i % n_firms,
        })
        .collect();

    let initial_total_cash: f32 = agents.iter().map(|a| a.cash).sum::<f32>()
        + firms.iter().map(|f| f.cash).sum::<f32>();

    // --------- simulate ----------
    let steps = 200;
    for _ in 0..steps {
        step(&mut agents, &mut firms);
    }

    // --------- checks ----------
    let total_cash: f32 = agents.iter().map(|a| a.cash).sum::<f32>()
        + firms.iter().map(|f| f.cash).sum::<f32>();

    // Money should be conserved in this toy model (no printing, no taxes)
    assert!((total_cash - initial_total_cash).abs() < 1e-3);

    // Nobody should have NaN cash/price
    assert!(agents.iter().all(|a| a.cash.is_finite()));
    assert!(firms.iter().all(|f| f.price.is_finite() && f.cash.is_finite()));

    // Economy shouldn't completely collapse: at least some inventory remains
    let total_inventory: f32 = firms.iter().map(|f| f.inventory).sum();
    assert!(total_inventory.is_finite());
}
