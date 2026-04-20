# SQL Insights

A few extra angles on the same tables that I think are worth looking at
given the "why are deliveries getting slower" question.

---

### Peak hours and where the delay comes from

When do deliveries actually slow down, and is it traffic or just volume?

```sql
SELECT EXTRACT(HOUR FROM order_placed_at) AS hour,
       COUNT(*)                           AS deliveries,
       AVG(delivery_time_min)             AS avg_time,
       AVG(delivery_distance_km)          AS avg_distance,
       AVG(CASE WHEN traffic_condition = 'High' THEN 1.0 ELSE 0 END) AS pct_high_traffic
FROM deliveries
GROUP BY 1
ORDER BY 1;
```

If the lunch / dinner peaks show both high avg time AND a jump in
`pct_high_traffic`, the slowdown is the city and the fix is staging couriers
near dense restaurant areas before the rush. If avg time spikes but traffic
doesn't, it's us — not enough couriers online, or kitchens overwhelmed — and
the fix is scheduling / surge incentives.

---

### Does courier experience actually help?

```sql
SELECT CASE
           WHEN (d.order_placed_at::date - dp.hired_date) < 30  THEN '<1m'
           WHEN (d.order_placed_at::date - dp.hired_date) < 90  THEN '1-3m'
           WHEN (d.order_placed_at::date - dp.hired_date) < 180 THEN '3-6m'
           WHEN (d.order_placed_at::date - dp.hired_date) < 365 THEN '6-12m'
           ELSE '12m+'
       END AS tenure,
       COUNT(*)               AS deliveries,
       AVG(d.delivery_rating) AS avg_rating,
       AVG(d.delivery_time_min) AS avg_time
FROM deliveries d
JOIN delivery_persons dp ON dp.delivery_person_id = d.delivery_person_id
GROUP BY 1;
```

If rating/time improves a lot from the <1m bucket to the 1-3m bucket and
then plateaus, most of the learning happens in the first weeks — worth
investing in a strong onboarding (shadow rides, mentor couriers) and not
overpaying for tenure. If the curve keeps climbing past 6 months, courier
retention is directly an SLA lever and churn should be treated as an Ops
problem, not just HR.

---

### Which areas get hit hardest by weather

For each restaurant area, compare avg time on clear days vs bad weather.

```sql
WITH by_area AS (
    SELECT restaurant_area, weather_condition, AVG(delivery_time_min) AS avg_time
    FROM deliveries
    GROUP BY restaurant_area, weather_condition
)
SELECT a.restaurant_area,
       b.weather_condition,
       a.avg_time AS clear_time,
       b.avg_time AS bad_time,
       b.avg_time - a.avg_time AS extra_minutes
FROM by_area a
JOIN by_area b USING (restaurant_area)
WHERE a.weather_condition = 'Clear'
  AND b.weather_condition <> 'Clear'
ORDER BY extra_minutes DESC;
```

Areas at the top of the list are the fragile ones — usually outskirts with
fewer active couriers, or areas with bad roads. Two concrete actions: (1)
widen the shown ETA in those areas when the forecast turns bad so CSAT
doesn't tank, and (2) pre-position or bonus couriers there ahead of the
storm. Areas where the extra is near zero don't need intervention, save the
budget.

---

### Which restaurants are actually slowing us down

Rank restaurants by how much worse their avg delivery time is compared to
other restaurants in the same area + cuisine.

```sql
WITH per_restaurant AS (
    SELECT r.restaurant_id, r.name, r.area, r.cuisine_type,
           r.avg_preparation_time_min,
           AVG(d.delivery_time_min) AS avg_time,
           COUNT(*) AS deliveries
    FROM restaurants r
    JOIN orders o     ON o.restaurant_id = r.restaurant_id
    JOIN deliveries d ON d.delivery_id = o.delivery_id
    GROUP BY r.restaurant_id, r.name, r.area, r.cuisine_type, r.avg_preparation_time_min
    HAVING COUNT(*) >= 30
),
peer AS (
    SELECT area, cuisine_type, AVG(avg_time) AS peer_avg
    FROM per_restaurant
    GROUP BY area, cuisine_type
)
SELECT pr.name, pr.area, pr.cuisine_type,
       pr.avg_preparation_time_min,
       pr.avg_time,
       pr.avg_time - p.peer_avg AS gap_vs_peers
FROM per_restaurant pr
JOIN peer p USING (area, cuisine_type)
ORDER BY gap_vs_peers DESC
LIMIT 20;
```

The important thing here is what the gap means when you cross it with prep
time. High gap + high `avg_preparation_time_min` = kitchen-bound; the
conversation is about prep time, throttling orders at peak, or changing their
SLA. High gap + normal prep = pickup friction (hard-to-find entrance, no
courier waiting area, bad handoff) — different fix, usually cheaper. Just
ranking "slowest restaurants" without that split sends Ops after the wrong
thing.
