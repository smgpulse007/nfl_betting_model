"""Fix Streamlit deprecation warnings"""
import re

files = [
    'task_8d2_model_performance.py',
    'task_8d3_feature_analysis.py',
    'task_8d4_betting_simulator.py'
]

for filename in files:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace use_container_width=True with width='stretch'
    content = re.sub(r'use_container_width=True', "width='stretch'", content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Fixed {filename}")

print("\n✅ All files fixed!")

