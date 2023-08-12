import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('asdf.csv')

# Avg compound score for reviews and headers by year
average_review_comp_score_by_year = df.groupby('year')['reviews_VADER_compound'].mean()
review_count_by_year = df['year'].value_counts().sort_index()
average_header_comp_score_by_year = df.groupby('year')['headers_VADER_compound'].mean()

# Create a figure and a single Axes for both plots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot average review compound scores as bars
average_review_comp_score_by_year.plot(kind='bar', ax=ax, color='blue', label='Avg Review Compound Score')

# Create a range of indices for the years
years_range = range(len(review_count_by_year))

# Plot review counts as a line on the same Axes
ax2 = ax.twinx()  # Create a twin Axes sharing the same x-axis
ax2.plot(years_range, review_count_by_year, color='black', marker='o', label='Review Count')

# Set x-axis ticks and labels
ax2.set_xticks(years_range)
ax2.set_xticklabels(review_count_by_year.index)

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Avg Review Compound Score', color='blue')
ax2.set_ylabel('Review Count', color='black')
ax.set_title('Average Review Compound Score and Review Count by Year')

# Combine the legends from both plots
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# Display the plot
plt.show()

# # Create a figure and a single Axes for both plots
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot average review compound scores as a line
# years_range = range(len(average_review_comp_score_by_year))
# ax.plot(years_range, average_review_comp_score_by_year, color='blue', marker='o', label='Avg Review Compound Score')

# # Set x-axis ticks and labels
# ax.set_xticks(years_range)
# ax.set_xticklabels(average_review_comp_score_by_year.index)

# # Plot review counts as bars on the same Axes
# ax2 = ax.twinx()  # Create a twin Axes sharing the same x-axis
# review_count_by_year.plot(kind='bar', ax=ax2, color='black', label='Review Count')

# # Set labels and title
# ax.set_xlabel('Year')
# ax.set_ylabel('Avg Review Compound Score', color='blue')
# ax2.set_ylabel('Review Count', color='black')
# ax.set_title('Average Review Compound Score and Review Count by Year')

# # Combine the legends from both plots
# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# # Display the plot
# plt.show()