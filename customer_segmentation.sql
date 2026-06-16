-- =====================================================================
-- customer_churn / sql / customer_segmentation.sql
-- =====================================================================
-- Segmentation queries — for marketing / retention team to action.
-- Uses RFM-style logic adapted for a subscription business.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Customer value segments
--
-- Combine tenure (loyalty) and monthly_charges (value) into a 3x3 grid:
--   - Loyalty: short / medium / long  (tenure)
--   - Value:   low    / medium / high (monthly_charges)
-- ---------------------------------------------------------------------
WITH tenure_quantiles AS (
    SELECT
        PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY tenure_months) AS p33_tenure,
        PERCENTILE_CONT(0.66) WITHIN GROUP (ORDER BY tenure_months) AS p66_tenure,
        PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY monthly_charges) AS p33_charges,
        PERCENTILE_CONT(0.66) WITHIN GROUP (ORDER BY monthly_charges) AS p66_charges
    FROM customers
),
segmented AS (
    SELECT
        c.customer_id,
        c.tenure_months,
        c.monthly_charges,
        c.churn,
        CASE
            WHEN c.tenure_months <= q.p33_tenure THEN 'Short'
            WHEN c.tenure_months <= q.p66_tenure THEN 'Medium'
            ELSE 'Long'
        END AS loyalty_segment,
        CASE
            WHEN c.monthly_charges <= q.p33_charges THEN 'Low'
            WHEN c.monthly_charges <= q.p66_charges THEN 'Medium'
            ELSE 'High'
        END AS value_segment
    FROM customers c CROSS JOIN tenure_quantiles q
)
SELECT
    loyalty_segment,
    value_segment,
    COUNT(*) AS n_customers,
    ROUND((AVG(CASE WHEN churn THEN 1.0 ELSE 0.0 END) * 100)::numeric, 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charges)::numeric, 2) AS monthly_revenue,
    ROUND((SUM(monthly_charges) * 12)::numeric, 2) AS annual_revenue
FROM segmented
GROUP BY loyalty_segment, value_segment
ORDER BY
    CASE loyalty_segment WHEN 'Short' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
    CASE value_segment   WHEN 'Low'   THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;


-- ---------------------------------------------------------------------
-- High-risk + high-value cohort — the priority intervention list
-- ---------------------------------------------------------------------
WITH latest AS (
    SELECT DISTINCT ON (customer_id)
        customer_id, predicted_prob, risk_tier
    FROM predictions
    ORDER BY customer_id, predicted_at DESC
),
charges_p75 AS (
    SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY monthly_charges) AS p75
    FROM customers
)
SELECT
    c.customer_id,
    c.tenure_months,
    c.contract,
    c.monthly_charges,
    l.predicted_prob,
    'priority' AS flag
FROM customers c
JOIN latest l       ON c.customer_id = l.customer_id
CROSS JOIN charges_p75 q
WHERE l.risk_tier = 'High'
  AND c.monthly_charges >= q.p75
ORDER BY l.predicted_prob DESC, c.monthly_charges DESC;
