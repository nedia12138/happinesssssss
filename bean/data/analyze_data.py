import pandas as pd
import numpy as np
import os

# 设置工作目录
os.chdir('')

# 读取数据
print('正在读取数据...')
try:
    train_abbr = pd.read_csv('happiness_train_abbr.csv', encoding='utf-8')
    test_abbr = pd.read_csv('happiness_test_abbr.csv', encoding='utf-8')

    # 尝试读取完整版数据
    try:
        train_complete = pd.read_csv('happiness_train_complete.csv', encoding='utf-8')
        test_complete = pd.read_csv('happiness_test_complete.csv', encoding='utf-8')
        complete_available = True
    except UnicodeDecodeError:
        print('完整版数据文件编码问题，跳过完整版数据分析')
        complete_available = False

except Exception as e:
    print(f'读取数据失败: {e}')
    exit(1)

print('=== 数据集概览 ===')
print(f'训练集_简化版: {train_abbr.shape[0]} 行, {train_abbr.shape[1]} 列')
print(f'测试集_简化版: {test_abbr.shape[0]} 行, {test_abbr.shape[1]} 列')

if complete_available:
    print(f'训练集_完整版: {train_complete.shape[0]} 行, {train_complete.shape[1]} 列')
    print(f'测试集_完整版: {test_complete.shape[0]} 行, {test_complete.shape[1]} 列')

print('\n=== 简化版数据列名 ===')
print('训练集列数:', len(train_abbr.columns))
print('列名:', list(train_abbr.columns))

if complete_available:
    print('\n=== 完整版额外列名 (相对于简化版) ===')
    abbr_cols = set(train_abbr.columns)
    complete_cols = set(train_complete.columns)
    extra_cols = complete_cols - abbr_cols
    print(f'完整版额外 {len(extra_cols)} 列:', sorted(list(extra_cols)))

print('\n=== 幸福感分布 (训练集简化版) ===')
happiness_dist = train_abbr['happiness'].value_counts().sort_index()
for level, count in happiness_dist.items():
    print(f'幸福感等级 {level}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 基本统计信息 ===')
print('年龄分布 (基于2015年):')
birth_year = 2015 - train_abbr['birth']
print(f'年龄范围: {birth_year.min()} - {birth_year.max()} 岁')
print(f'平均年龄: {birth_year.mean():.1f} 岁')
print(f'中位年龄: {birth_year.median():.1f} 岁')

print('\n收入分布 (元):')
valid_income = train_abbr['income'][train_abbr['income'] > 0]
print(f'有效收入样本数: {len(valid_income)}')
print(f'平均收入: {valid_income.mean():,.0f} 元')
print(f'中位收入: {valid_income.median():,.0f} 元')
print(f'收入标准差: {valid_income.std():,.0f} 元')

print('\n=== 数据质量检查 ===')
print('缺失值统计:')
missing_train = train_abbr.isnull().sum()
missing_cols = missing_train[missing_train > 0]
if len(missing_cols) > 0:
    print('存在缺失值的列:')
    for col, count in missing_cols.items():
        print(f'{col}: {count} 缺失 ({count/train_abbr.shape[0]*100:.1f}%)')
else:
    print('无缺失值')

print('\n=== 性别分布 ===')
gender_dist = train_abbr['gender'].value_counts()
gender_map = {1: '男', 2: '女'}
for gender_code, count in gender_dist.items():
    gender_name = gender_map.get(gender_code, f'未知({gender_code})')
    print(f'{gender_name}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 省份分布 (前10个) ===')
province_dist = train_abbr['province'].value_counts().head(10)
for province, count in province_dist.items():
    print(f'省份 {province}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 教育水平分布 ===')
edu_dist = train_abbr['edu'].value_counts().sort_index()
edu_map = {
    1: '文盲', 2: '小学', 3: '初中', 4: '高中', 5: '大专', 6: '本科',
    7: '硕士', 8: '博士', 9: '其他', 10: '初中以下', 11: '高中以下', 12: '大专以下'
}
for edu_code, count in edu_dist.items():
    edu_name = edu_map.get(edu_code, f'未知({edu_code})')
    print(f'{edu_name}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 健康状况分布 ===')
health_dist = train_abbr['health'].value_counts().sort_index()
health_map = {1: '非常健康', 2: '比较健康', 3: '一般', 4: '不太健康', 5: '不健康'}
for health_code, count in health_dist.items():
    health_name = health_map.get(health_code, f'未知({health_code})')
    print(f'{health_name}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 婚姻状况分析 ===')
marital_dist = train_abbr['marital'].value_counts().sort_index()
marital_map = {
    1: '未婚', 2: '同居', 3: '初婚有配偶', 4: '再婚有配偶',
    5: '离婚', 6: '丧偶', 7: '其他'
}
for marital_code, count in marital_dist.items():
    marital_name = marital_map.get(marital_code, f'未知({marital_code})')
    print(f'{marital_name}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 教育水平详细分析 ===')
edu_dist = train_abbr['edu'].value_counts().sort_index()
edu_map = {
    1: '文盲', 2: '小学', 3: '初中', 4: '高中', 5: '大专', 6: '本科',
    7: '硕士', 8: '博士', 9: '其他', 10: '初中以下', 11: '高中以下',
    12: '大专以下', 13: '未知1', 14: '未知2'
}
print('教育水平分布:')
for edu_code, count in edu_dist.items():
    edu_name = edu_map.get(edu_code, f'未知({edu_code})')
    print(f'{edu_name}: {count} 人 ({count/train_abbr.shape[0]*100:.1f}%)')

# 按教育程度分组统计
print('\n教育程度大类统计:')
high_edu = edu_dist[edu_dist.index >= 6].sum()  # 本科及以上
middle_edu = edu_dist[(edu_dist.index >= 4) & (edu_dist.index <= 5)].sum()  # 高中-大专
low_edu = edu_dist[edu_dist.index <= 3].sum()  # 初中及以下

print(f'高等教育 (本科及以上): {high_edu} 人 ({high_edu/train_abbr.shape[0]*100:.1f}%)')
print(f'中等教育 (高中-大专): {middle_edu} 人 ({middle_edu/train_abbr.shape[0]*100:.1f}%)')
print(f'基础教育 (初中及以下): {low_edu} 人 ({low_edu/train_abbr.shape[0]*100:.1f}%)')

print('\n=== 收入区间分析 ===')
# 过滤有效收入数据 (去除负值)
valid_income = train_abbr['income'][train_abbr['income'] > 0]
print(f'有效收入样本数: {len(valid_income)}')
print(f'收入统计 (元):')
print(f'最小值: {valid_income.min():,.0f}')
print(f'最大值: {valid_income.max():,.0f}')
print(f'平均值: {valid_income.mean():,.0f}')
print(f'中位数: {valid_income.median():,.0f}')

# 收入区间分组
print('\n收入区间分布 (元/年):')
income_bins = [0, 10000, 30000, 50000, 100000, 200000, float('inf')]
income_labels = ['1万以下', '1-3万', '3-5万', '5-10万', '10-20万', '20万以上']
income_groups = pd.cut(valid_income, bins=income_bins, labels=income_labels, right=False)

income_dist = income_groups.value_counts().sort_index()
for income_range, count in income_dist.items():
    print(f'{income_range}: {count} 人 ({count/len(valid_income)*100:.1f}%)')

print('\n=== 关键特征与幸福感的相关性分析 ===')

# 婚姻状况与幸福感
print('婚姻状况 vs 幸福感:')
marital_happiness = pd.crosstab(train_abbr['marital'], train_abbr['happiness'], normalize='index') * 100
for marital_code in sorted(train_abbr['marital'].unique()):
    if marital_code in marital_map:
        print(f"\n{marital_map[marital_code]}:")
        happy_dist = marital_happiness.loc[marital_code]
        for happiness_level in [1,2,3,4,5]:
            if happiness_level in happy_dist.index:
                print(f"  幸福感{happiness_level}: {happy_dist[happiness_level]:.1f}%")

# 教育水平与幸福感 (合并相似学历)
edu_happiness = train_abbr.copy()
edu_happiness['edu_group'] = pd.cut(edu_happiness['edu'],
                                   bins=[0, 3, 5, float('inf')],
                                   labels=['初中及以下', '高中-大专', '本科及以上'])
edu_group_happiness = pd.crosstab(edu_happiness['edu_group'], edu_happiness['happiness'], normalize='index') * 100

print('\n教育水平 vs 幸福感:')
for edu_group in ['初中及以下', '高中-大专', '本科及以上']:
    if edu_group in edu_group_happiness.index:
        print(f"\n{edu_group}:")
        happy_dist = edu_group_happiness.loc[edu_group]
        avg_happiness = sum(level * pct/100 for level, pct in happy_dist.items() if level in [1,2,3,4,5])
        print(f"  平均幸福感: {avg_happiness:.2f}")
        for happiness_level in [1,2,3,4,5]:
            if happiness_level in happy_dist.index:
                print(f"  幸福感{happiness_level}: {happy_dist[happiness_level]:.1f}%")

# 收入区间与幸福感
income_happiness = train_abbr[train_abbr['income'] > 0].copy()
income_happiness['income_group'] = pd.cut(income_happiness['income'],
                                         bins=[0, 30000, 100000, float('inf')],
                                         labels=['3万以下', '3-10万', '10万以上'])
income_group_happiness = pd.crosstab(income_happiness['income_group'], income_happiness['happiness'], normalize='index') * 100

print('\n收入区间 vs 幸福感:')
for income_group in ['3万以下', '3-10万', '10万以上']:
    if income_group in income_group_happiness.index:
        print(f"\n{income_group}:")
        happy_dist = income_group_happiness.loc[income_group]
        avg_happiness = sum(level * pct/100 for level, pct in happy_dist.items() if level in [1,2,3,4,5])
        print(f"  平均幸福感: {avg_happiness:.2f}")
        for happiness_level in [1,2,3,4,5]:
            if happiness_level in happy_dist.index:
                print(f"  幸福感{happiness_level}: {happy_dist[happiness_level]:.1f}%")
