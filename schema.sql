-- =====================================================================
-- customer_churn / sql / schema.sql
-- =====================================================================
-- Schema for the churn warehouse. One central customer table, plus
-- a predictions table to hold model output by run.
-- =====================================================================

DROP TABLE IF EXISTS predictions     CASCADE;
DROP TABLE IF EXISTS customers       CASCADE;
DROP TABLE IF EXISTS service_options CASCADE;


-- ---------------------------------------------------------------------
-- customers — one row per customer with their service mix + outcome
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id          VARCHAR(20) PRIMARY KEY,
    gender               VARCHAR(10),
    senior_citizen       BOOLEAN,
    partner              BOOLEAN,
    dependents           BOOLEAN,
    tenure_months        INTEGER,
    phone_service        BOOLEAN,
    multiple_lines       VARCHAR(20),
    internet_service     VARCHAR(20),
    online_security      VARCHAR(20),
    online_backup        VARCHAR(20),
    device_protection    VARCHAR(20),
    tech_support         VARCHAR(20),
    streaming_tv         VARCHAR(20),
    streaming_movies     VARCHAR(20),
    contract             VARCHAR(20),
    paperless_billing    BOOLEAN,
    payment_method       VARCHAR(40),
    monthly_charges      NUMERIC(10, 2),
    total_charges        NUMERIC(12, 2),
    churn                BOOLEAN
);

CREATE INDEX idx_customers_contract ON customers(contract);
CREATE INDEX idx_customers_tenure ON customers(tenure_months);
CREATE INDEX idx_customers_churn ON customers(churn);


-- ---------------------------------------------------------------------
-- predictions — model output per customer per run
-- ---------------------------------------------------------------------
CREATE TABLE predictions (
    prediction_id        BIGSERIAL PRIMARY KEY,
    customer_id          VARCHAR(20) NOT NULL REFERENCES customers(customer_id),
    model_version        VARCHAR(20),
    predicted_prob       NUMERIC(6, 5),         -- 0.0000 to 1.0000
    risk_tier            VARCHAR(10),           -- 'High' / 'Medium' / 'Low'
    predicted_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_customer ON predictions(customer_id);
CREATE INDEX idx_predictions_tier ON predictions(risk_tier);
CREATE INDEX idx_predictions_at ON predictions(predicted_at);
