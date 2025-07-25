🏨 Hotel Booking Data EDA
📌 Problem Statement
Uncover patterns in hotel booking data to support operational efficiency and revenue growth.

📊 Project Overview
This exploratory data analysis (EDA) investigates a hotel bookings dataset to:
Understand customer booking behavior and demographics.
Analyze pricing (ADR), stay patterns, cancellations, and booking channels.
Detect inconsistencies in room assignments and guest handling.
Validate key business assumptions through statistical testing.

🎯 Core Objectives
Identify trends in lead time, stay duration, and market segments.
Analyze ADR (Average Daily Rate) variation across channels and guest types.
Detect operational anomalies in room allocation and special requests.
Use hypothesis testing to validate insights on pricing, upgrades, and guest behavior.

🧹 Data Cleaning & Preprocessing
Dataset: 119,390 rows and 32 columns.
Removed 31,994 duplicate rows → 87,396 records used.

Missing value treatment:
Mode imputation for children, country, and agent.
Dropped company column due to >93% missing values.
Outlier capping (IQR method) applied to lead_time and adr.

🧠 Feature Engineering
Classified variables into categorical, discrete, and continuous.
Created derived features like stay duration and total guests.
Used value counts and distribution plots to guide transformations.

📈 Exploratory Data Analysis (EDA)
Univariate analysis (histograms) for variables like adr, lead_time, market_segment.
Bivariate analysis:
Boxplots for adr across market segments.
Heatmap to examine correlation matrix.
Time-series analysis to identify seasonal trends in bookings.

🔍 Key Insights
Online Travel Agencies (OTA) dominate bookings.
ADR varies significantly by market segment and distribution channel.
Longer lead time → more booking modifications.
Guests with higher ADR tend to request more special services.
Booking behavior varies by country of origin.

📊 Hypothesis Testing
ADR: OTA vs Direct Bookings → Significant difference in pricing.
Room Upgrades vs Lead Time → No significant relationship.
Stay Duration vs Customer Type → Significant variation found.

📌 Business Insights
Transient customers have the highest ADR.
Booking channel plays a crucial role in revenue strategy.
Longer lead times may require better follow-up or modification handling.
Certain countries yield higher ADR, useful for targeted marketing.

✅ Conclusion
The project highlights actionable insights for hotel managers to:
Optimize room pricing strategies.
Focus marketing on high-revenue segments.
Improve operational planning around customer preferences and booking behavior.



