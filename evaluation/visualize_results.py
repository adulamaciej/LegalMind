import matplotlib.pyplot as plt

variants = ['A (with debate)', 'B (no debate)']
exact_match = [66.7, 66.7]
partial_match = [100.0, 100.0]

x = range(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([i - width/2 for i in x], exact_match, width, label='Exact match', color='steelblue')
ax.bar([i + width/2 for i in x], partial_match, width, label='Partial match', color='lightblue')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Debate vs No-Debate: Verdict Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.legend()

plt.tight_layout()
plt.savefig('evaluation_results.png')
plt.show()