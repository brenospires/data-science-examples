# Demand Forecasting and Inventory Optimization Under Censored Demand

## Project Overview

In real-world supply chain environments, demand forecasting models are often trained using historical sales data. However, sales are not always an accurate representation of true customer demand. During stockout periods, sales become censored because customers cannot purchase products that are unavailable, causing historical data to systematically underestimate actual demand.

The objective of this project is to build an end-to-end supply chain optimization framework capable of operating under censored demand conditions. The project begins with the generation of a realistic synthetic retail dataset containing multiple demand behaviors, inventory dynamics, replenishment policies, and frequent stockout events. This environment allows the evaluation of forecasting and inventory management strategies in a controlled setting where the true demand is known.

Using the simulated data, products are segmented according to their demand patterns through unsupervised learning techniques. Dedicated forecasting models are then developed for each segment, enabling the system to capture the unique characteristics of different demand profiles. Forecast uncertainty is incorporated into the decision-making process, allowing inventory policies to be defined based not only on expected demand but also on the associated risk.

The final stage of the project consists of a business simulation that compares the original inventory policy against a new policy driven by machine learning forecasts. Both approaches are evaluated using operational and financial metrics such as inventory holding costs, stockout costs, service levels, fill rates, and total supply chain cost.

The ultimate goal is to demonstrate how data science can be used to improve inventory decisions, reduce operational costs, increase product availability, and generate measurable business value in supply chain planning scenarios.
