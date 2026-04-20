-- Food Delivery Intelligence - Part I (PostgreSQL)


-- Q1. Top 5 customer areas with highest avg delivery time, last 30 days.
SELECT customer_area,
       AVG(delivery_time_min) AS avg_time,
       COUNT(*) AS deliveries
FROM deliveries
WHERE order_placed_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY customer_area
ORDER BY avg_time DESC
LIMIT 5;


-- Q2. Avg delivery time per traffic condition, by restaurant area and cuisine.
SELECT r.area,
       r.cuisine_type,
       d.traffic_condition,
       AVG(d.delivery_time_min) AS avg_time,
       COUNT(*) AS deliveries
FROM deliveries d
JOIN orders o      ON o.delivery_id = d.delivery_id
JOIN restaurants r ON r.restaurant_id = o.restaurant_id
GROUP BY r.area, r.cuisine_type, d.traffic_condition
ORDER BY r.area, r.cuisine_type, avg_time DESC;


-- Q3. Top 10 fastest delivery persons (still active, at least 50 deliveries).
SELECT dp.delivery_person_id,
       dp.name,
       dp.region,
       COUNT(*) AS deliveries,
       AVG(d.delivery_time_min) AS avg_time
FROM deliveries d
JOIN delivery_persons dp ON dp.delivery_person_id = d.delivery_person_id
WHERE dp.is_active
GROUP BY dp.delivery_person_id, dp.name, dp.region
HAVING COUNT(*) >= 50
ORDER BY avg_time
LIMIT 10;


-- Q4. Most profitable restaurant area in the last 3 months.
SELECT r.area,
       SUM(o.order_value) AS total_value
FROM orders o
JOIN deliveries d  ON d.delivery_id = o.delivery_id
JOIN restaurants r ON r.restaurant_id = o.restaurant_id
WHERE d.order_placed_at >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY r.area
ORDER BY total_value DESC
LIMIT 1;


-- Q5. Delivery persons with an increasing trend in avg delivery time.
-- Monthly avg per person, compare vs previous month with LAG.
-- Flag the ones where every month-over-month change went up (need >= 3 months).
WITH monthly AS (
    SELECT delivery_person_id,
           DATE_TRUNC('month', order_placed_at) AS month,
           AVG(delivery_time_min) AS avg_time
    FROM deliveries
    GROUP BY delivery_person_id, DATE_TRUNC('month', order_placed_at)
),
with_lag AS (
    SELECT delivery_person_id,
           month,
           avg_time,
           LAG(avg_time) OVER (PARTITION BY delivery_person_id ORDER BY month) AS prev_avg
    FROM monthly
)
SELECT delivery_person_id,
       COUNT(*) AS months,
       MIN(avg_time) AS min_avg,
       MAX(avg_time) AS max_avg
FROM with_lag
GROUP BY delivery_person_id
HAVING COUNT(*) >= 3
   AND COUNT(prev_avg) = SUM(CASE WHEN avg_time > prev_avg THEN 1 ELSE 0 END)
ORDER BY max_avg - min_avg DESC;
