# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习工具
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 数据加载
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# 探索性数据分析（EDA）
def basic_eda(df):
    print("数据概览:")
    print(df.info())
    print("\n缺失值统计:")
    print(df.isnull().sum())
    print("\n数值特征统计:")
    print(df.describe())


basic_eda(train_df)

# 可视化分析
plt.figure(figsize=(12, 6))
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('不同舱位等级的生存情况')
plt.show()


# 特征工程
def feature_engineering(df):
    # 提取称呼（Mr, Mrs等）
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # 处理客舱信息
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'Unknown')

    # 票价分箱
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

    # 年龄分箱
    df['AgeBin'] = pd.cut(df['Age'].fillna(df['Age'].median()),
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=[0, 1, 2, 3, 4])
    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# 定义预处理管道
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'AgeBin', 'FareBin']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# 构建基础模型管道
base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# 划分数据集
X = train_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 交叉验证评估
print("\n交叉验证评估：")
cv_scores = cross_val_score(base_model, X, y, cv=5)
print(f"交叉验证平均准确率: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

# 超参数调优
print("\n超参数调优中...")
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5]
}

grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳验证分数: {grid_search.best_score_:.2f}")

# 使用最佳模型
best_model = grid_search.best_estimator_

# 验证集评估
y_pred = best_model.predict(X_val)
print("\n最佳模型验证集表现：")
print(f"准确率: {accuracy_score(y_val, y_pred):.2f}")
print(classification_report(y_val, y_pred))

# 构建集成模型
print("\n构建集成模型...")
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

ensemble = VotingClassifier(
    estimators=[
        ('best_gb', best_model),
        ('rf', rf_model)
    ],
    voting='soft')

ensemble.fit(X_train, y_train)

# 集成模型评估
ensemble_pred = ensemble.predict(X_val)
print("\n集成模型验证集表现：")
print(f"准确率: {accuracy_score(y_val, ensemble_pred):.2f}")
print(classification_report(y_val, ensemble_pred))

# 特征重要性分析（使用最佳单一模型）
feature_importance = best_model.named_steps['classifier'].feature_importances_
onehot_columns = best_model.named_steps['preprocessor'].named_transformers_['cat'] \
    .named_steps['onehot'].get_feature_names_out(categorical_features)
all_features = numeric_features + list(onehot_columns)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance, y=all_features)
plt.title('特征重要性排序')
plt.show()

# 生成最终预测（使用集成模型）
test_X = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
final_pred = ensemble.predict(test_X)

# 保存提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_pred
})
submission.to_csv('submission.csv', index=False)

print("\n提交文件已保存为 submission.csv")