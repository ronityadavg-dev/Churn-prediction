-- =====================================================================
-- customer_churn / sql / churn_kpis.sql
-- =====================================================================
-- The queries powering the churn dashboard.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Q1: headline KPIs
-- ---------------------------------------------------------------------
SELECT
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)              AS total_churned,
    AVG(CASE WHEN churn THEN 1.0 ELSE 0.0 END) * 100    AS churn_rate_pct,
    SUM(monthly_charges)                                AS monthly_revenue,
    SUM(monthly_charges) * 12                           AS annual_revenue,
    SUM(CASE WHEN churn THEN monthly_charges ELSE 0 END) * 12 AS revenue_lost_annual
FROM customers;


-- ---------------------------------------------------------------------
-- Q2: churn rate by contract — biggest single driver in this data
-- ---------------------------------------------------------------------
SELECT
    contract,
    COUNT(*)                                                      AS n,
    SUM(CASE WHEN churn THEN 1 ELSE 0 END)                        AS n_churned,
    ROUND((AVG(CASE WHEN churn THEN 1.0 ELSE 0.0 END) * 100)::numeric, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges)::numeric, 2)                       AS avg_monthly_charge
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q3: churn by tenure bucket — survival curve in table form
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN tenure_months <= 6   THEN '0-6m'
        WHEN tenure_months <= 12  THEN '6-12m'
        WHEN tenure_months <= 24  THEN '1-2y'
        WHEN tenure_months <= 48  THEN '2-4y'
        WHEN tenure_months <= 72  THEN '4-6y'
        ELSE '6y+'
    END AS tenure_bucket,
    COUNT(*) AS n,
    ROUND((AVG(CASE WHEN churn THEN 1.0 ELSE 0.0 END) * 100)::numeric, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges)::numeric, 2) AS avg_monthly_charge
FROM customers
GROUP BY tenure_bucket
ORDER BY
    CASE tenure_bucket
        WHEN '0-6m'  THEN 1
        WHEN '6-12m' THEN 2
        WHEN '1-2y'  THEN 3
        WHEN '2-4y'  THEN 4
        WHEN '4-6y'  THEN 5
        WHEN '6y+'   THEN 6
    END;


-- ---------------------------------------------------------------------
-- Q4: payment method × internet service — find the toxic combinations
-- ---------------------------------------------------------------------
SELECT
    payment_method,
    internet_service,
    COUNT(*) AS n,
    ROUND((AVG(CASE WHEN churn THEN 1.0 ELSE 0.0 END) * 100)::numeric, 2) AS churn_rate_pct
FROM customers
GROUP BY payment_method, internet_service
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q5: risk tier breakdown from the latest predictions
-- ---------------------------------------------------------------------
WITH latest AS (
    SELECT DISTINCT ON (customer_id)
        customer_id, predicted_prob, risk_tier, predicted_at
    FROM predictions
    ORDER BY customer_id, predicted_at DESC
)
SELECT
    l.risk_tier,
    COUNT(*)                                              AS n_customers,
    ROUND(AVG(l.predicted_prob)::numeric, 3)              AS avg_prob,
    ROUND(SUM(c.monthly_charges)::numeric, 2)             AS monthly_at_risk,
    ROUND((SUM(c.monthly_charges) * 12)::numeric, 2)      AS annual_at_risk
FROM latest l
JOIN customers c ON l.customer_id = c.customer_id
GROUP BY l.risk_tier
ORDER BY
    CASE l.risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;


-- ---------------------------------------------------------------------
-- Q6: top 100 highest-risk customers — for the call list
-- ---------------------------------------------------------------------
WITH latest AS (
    SELECT DISTINCT ON (customer_id)
        customer_id, predicted_prob, risk_tier
    FROM predictions
    ORDER BY customer_id, predicted_at DESC
)
SELECT
    c.customer_id,
    c.tenure_months,
    c.contract,
    c.payment_method,
    c.monthly_charges,
    c.total_charges,
    l.predicted_prob,
    l.risk_tier
FROM latest l
JOIN customers c ON l.customer_id = c.customer_id
WHERE l.risk_tier = 'High'
ORDER BY l.predicted_prob DESC, c.monthly_charges DESC
LIMIT 100;


-- ---------------------------------------------------------------------
-- Q7: services breakdown — does having more add-ons protect against churn?
-- ---------------------------------------------------------------------
WITH service_counts AS (
    SELECT
        customer_id,
        churn,
        monthly_charges,
        (CASE WHEN online_security  = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN online_backup    = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN device_protection= 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN tech_support     = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN streaming_tv     = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END)
        AS n_addons
    FROM customers
)
SELECT
    n_addons,
    COUNT(*) AS n,
    ROUND((AVG(CASE WHEN churn THEN 1.0 ELSE 0.0 END) * 100)::numeric, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges)::numeric, 2) AS avg_monthly
FROM service_counts
GROUP BY n_addons
ORDER BY n_addons;
