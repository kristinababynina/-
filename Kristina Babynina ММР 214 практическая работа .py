#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', sep = ',', index_col = 0)


# In[4]:


df.head ()


# Проверим датасет, посмотрим на наличие пропусков, изучим типы данных в столбцах

# In[5]:


rows, columns = df.shape
print(f'Строк: {rows} \nСтолбцов: {columns}')


# In[6]:


df.info () #применили метод инфо


# **Целочисленные столбцы:**
# 
# + ClothingID - ID одежды
# + Age - Возраст
# + Rating - рейтинг, выставленный клиентом, от 1 до 5
# + Recommended IND - Бинарная переменная, готовность клиента рекоммендовать товар
# + Positive Feedback Count - Количество лайков под комментарием
# 
# **Столбцы с текстом (object):**
# 
# + Title - заголовок комментария
# + Review Text - текст комментария
# + Division Name - имя высокоуровневого дивизиона, к которому относится товар
# + Department Name - имя подразделения, к которому относится товар (вверх, низ, топы, платья)
# + Class Name - класс продукта или конкретный "тип" одежды (штаны, кофта и т.п.)

# Исходя из текущих данных, можно выдвинуть следующие направления анализа данных:
# + Выявить средние значения рейтинга в разрезе по типам одежды, департаментам и дивизионам
# + Выявить влияние рейтинга на готовность покупателей рекомендовать товар (recommended_ind)
# + Выявить средние значения возраста по типам одежды
# + Оценить среднее кол-во комментариев на один конкретный товар

# Видим пропуски в заголовке, в тексте комментария, в категориях одежды. Данных много, поэтому можно просто избавиться от пропусков в категориях.

# Также для удобства работы со столбцами, сделаем названия более удобными

# In[8]:


df.columns


# In[9]:


new_columns = df.columns.str.lower().str.replace(' ','_') #изменили названия колонок для удобства
old_columns = df.columns

column_names = {}
for i in range(0,len(old_columns)):
    column_names[old_columns[i]] = new_columns[i] #создали словарь со старыми и новыми названиями
column_names


# In[10]:


df = df.rename(columns = column_names) #переименовали колонки в нашем датасете
df.head() 


# Теперь уберем пропуски

# In[11]:


df = df.dropna(axis = 0, subset  = ['division_name','department_name','class_name'])
#убрали пропуски только по этим колонкам, поскольку пока title и review_text не дадут полезной информации


# In[12]:


df.info ()


# Пропуски остались в комментариях, но для анализа они пока не понадобятся, поэтому их можно не трогать

# Посмотрим также на уникальные значения в категориальных переменных и их частоту

# In[13]:


df.division_name.value_counts()


# Наиболее часто встречающийся дивизион - General (одежда обычных размеров)

# In[14]:


df.department_name.value_counts()


# Видим, что меньше всего комментариев к одежде департамента Trend. Наибольшее кол-во в Tops. 
# Также на втором месте по частоте встречаются платья (dresses). Далее можно сравнить суммарное кол-во комментариев по tops и bottoms.

# In[15]:


df.class_name.value_counts()


# Наиболее популярный тип одежды в комментариях - платья

# In[16]:


df['class_name'].nunique()


# В датасете встречаются 20 уникальных типов одежды

# # Анализ количественных фичей

# In[17]:


quantative_columns = ['age', 'rating', 'recommended_ind', 'positive_feedback_count'] 
#создадим отдельный список количесвтенных фичей


# In[18]:


df[quantative_columns].describe()


# Средний возраст комментирующих 43 
# Средний рейтинг всех комментариев одежды - 4.19 (пока не совсем релевантно, поскольку одежда может повторяться) 
# В среднем комментирующие склоняются к тому, чтобы рекоммендовать приобретенный товар (среднее 0.8) 
# Среднее кол-во лайков под комментариями - 2.53 (максимально 122) 
# Более 50% оценок - 5 баллов. Это означает, что чаще всего люди оставляют отзыв на товар, когда он им понравился.

# ### Посмотрим на распределения возраста комментирующих

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


plt.figure(figsize = (10,5))

sns.histplot(x = 'age', data = df, bins = 20, color = 'green')     .set(title='Распределение возраста',
        xlabel = 'Возраст',
        ylabel = 'Количество')


# Распределение возраста немного похоже на нормальное, возможно максимальный возраст 99, это выброс. 
# <br> Чаще всего комментируют женщины в возрасте от 30 до 40.

# In[23]:


plt.figure(figsize = (4,6))
sns.boxplot(y = 'age', data = df, color = 'green').set(title='Распределение возраста', ylabel = 'Возраст')


# Боксплот отметил возраст выше 80 в качестве выбросов. Также у возраста довольно сильный разброс

# In[24]:


df ['age'].mean()


# In[27]:


import scipy.stats
scipy.stats.zscore (df['age']).hist()


# In[42]:


(df['age'].mean()-43)/df['age'].std()


# Среднее значение Z оценок близко к нулю, что свидетельствует о том что средний возраст действительно равен 43.

# Посмотрим на возраст в разрезе дивизионов и типов одежды. Можем определить средний возраст по каждой категории.

# In[29]:


age_by_div_class = df.groupby(['class_name'],as_index = False)['age'].mean().sort_values('age', ascending = False)


# In[30]:


age_by_class = df.groupby(['class_name'])['age'].mean().reset_index().sort_values('age', ascending = False) #группируем

plt.figure(figsize = (18,5)) 
plt.xticks(rotation=45) #поворот подписей типов
ax = sns.barplot(x = 'class_name', y = 'age', data = age_by_class, color = 'green')     .set(title='Распределение возраста по типам одежды',
            xlabel = 'Тип одежды',
            ylabel = 'Возраст') #задаем названия осей и графика


# Наибольший средний возраст среди покупателей свитеров, наименьший в домашней/полуспортивной одежде для низа. (casual bottoms)

# In[31]:


age_by_div_class = df.groupby(['division_name'])['age'].mean().reset_index()

plt.figure(figsize = (5,5))
plt.xticks(rotation=45) #поворот подписей типов
ax = sns.barplot(x = 'division_name', y = 'age', data = age_by_div_class, color = 'green')     .set(title='Распределение возраста по типам одежды',
            xlabel = 'Тип одежды',
            ylabel = 'Возраст') #задаем названия осей и графика


# Мы видим, что по дивизионам различия в среднем возрасте небольшие

# ### Распределение рейтинга 

# In[32]:


plt.figure(figsize = (10,5))
sns.histplot(x='rating', data=df, bins=5, binwidth=0.25, color = 'green')     .set(title='Распределение рейтинга',
        xlabel = 'Рейтинг',
        ylabel = 'Количество')


# In[33]:


rating_freq = df.groupby(['rating', 'recommended_ind'])     .agg({'rating':'count'})     .rename(columns = {'rating':'count'})     .reset_index()

rating_freq


# In[34]:


plt.figure(figsize = (10,5))
cat = sns.catplot(
    data=rating_freq, kind="bar",
    x="rating", y="count", hue="recommended_ind", palette='Blues', legend = False) \
    .set(title='Распределение рейтинга',
        xlabel = 'Рейтинг',
        ylabel = 'Количество')
cat.add_legend(title = 'Рекомендую этот товар')


# Как видно из графика, для высоких оценок 4 и 5 большее кол-во комментариев сопровождалось готовностью клиента рекомендовать товар. 
# При этом в комментариях с низким рейтингом стоит больше 0 в готовности рекомендовать. 
# Можно выдвинуть гипотезу о том, что рейтинг и готовность рекомендовать имеют положительную корреляцию: чем больше рейтинг, тем более вероятно, что товар порекомендуют

# In[35]:


df['recommended_ind'].corr (df['rating']) #используем корреляцию Пирсона


# Корреляция Пирсона показала высокую зависимость рекомендации от рейтинга

# Посмотрим на средние значения оценок по типам одежды

# In[36]:


class_rating = df.groupby('class_name')['rating'].mean().reset_index().sort_values('rating',ascending = False)
class_rating['rating'] = class_rating['rating'].round(2)
class_rating


# In[37]:


#https://plotly.com/python/bar-charts/
#используем plotly express для удобного интерактивного графика
fig = px.bar(class_rating, x='class_name', y='rating')
fig.show()


# Наибольший средний рейтинг в категории Casual Bottoms, наименьший в категории трендовых вещей и chemises (рубашки, ночнушки)

# In[39]:


department_rating = df.groupby('department_name')['rating'].mean().reset_index()
department_rating['rating'] = department_rating['rating'].round(2)
department_rating


# In[40]:


fig = px.bar (department_rating, x='department_name', y='rating')
fig.show ()


# В разрезе отделов одежды также видим более низкий средний рейтинг в категории Trend. Однако, важно учитывать, что у этой категории наименьшее кол-во комментариев в датасете.

# ### Распределение рекоммендаций (recommended_ind)

# In[44]:


plt.figure(figsize = (10,5))
sns.histplot(x='recommended_ind', data=df, binwidth=0.5)


# Большая часть комментариев сопровождается готовностью клиента рекомендовать товар.

# Проверим распределение оценок и готовность рекомендовать по возрасту. Для этого создадим отдельный столбец с интервальным возрастом.

# In[66]:


#для разбивки будем брать квартили
df['age'].describe()


# In[46]:


#зададим функцию для создания нового столбца с интервальным возрастом
def get_age_cat(age):
    if age <= 34:
        return 'Young'
    elif age <= 41:
        return 'Millennials'
    elif age <= 52:
        return 'GenX'
    elif age > 52:
        return 'Old'


# In[47]:


df['cat_age'] = pd.Categorical(df['age'])


# In[48]:


df['cat_age'] = df['cat_age'].apply(get_age_cat)


# In[49]:


df.head ()


# In[50]:


df['cat_age'].value_counts ()


# In[51]:


rating_by_age = df.groupby(['rating','cat_age'])['clothing_id'].count()     .reset_index()     .rename(columns = {'clothing_id':'count'})
rating_by_age


# In[52]:


plt.figure(figsize = (10,5))
ax = sns.catplot(
    data=rating_by_age, kind="bar",
    x="rating", y="count", hue="cat_age", palette='Blues', legend = False) \
    .set(title='Распределение рейтинга в разрезе по возрастным категориям',
        xlabel = 'Рейтинг',
        ylabel = 'Количество')
ax.add_legend(title = 'Возраст')


# Как видно из графика, сложно выявить взаимосвязь возрастных групп и рейтинга. 
# Можно заметить, что группа old (старше 52 лет) реже остальных ставит оценки 2,3,4. Однако важно понимать, что людей данной категории в наших данных меньше остальных (видим из value_counts выше)

# In[54]:


df.groupby(['cat_age'])['rating'].mean()     .reset_index()     .rename(columns = {'rating':'mean_rating'})     .sort_values('mean_rating',ascending = False)


# Если рассмотреть средний ретинг по возрастным категориям, то у категории old как раз наиболее высокий средний рейтинг

# ### Распределение лайков под комментариями

# In[55]:


plt.figure(figsize = (10,5))
sns.histplot(x='positive_feedback_count', data=df,bins = 20)

df['positive_feedback_count'].describe() #Ещё раз посмотрим на меры центральной тенденции
df['positive_feedback_count'].quantile(0.9)


# In[56]:


df['positive_feedback_count'].describe() #Ещё раз посмотрим на меры центральной тенденции


# Поскольку, 75% всех комментариев не набирает больше 30 лайков, построим боксплот для значений меньше 90% квантиля

# In[57]:


df['positive_feedback_count'].quantile(0.9)#возьмемь квантиль 90%, чтобы отсечь "выбросы"


# In[58]:


df[df.positive_feedback_count <= 7]['positive_feedback_count'].describe()
#если взять кол-во лайков меньше 7, то половина всех комментариев имеет 0 лайков.


# По медиане видно, что половина комментариев остаются без лайков

# In[59]:


plt.figure(figsize = (4,6))
sns.boxplot(y = 'positive_feedback_count', data = df[df.positive_feedback_count <= 7])


# Как видно из графиков и квартилей, лайков под комментариями действительно очень мало. Это может быть связано с тем, что пользователи редко оценивают отзывы друг друга.

# ### Количество комментариев в разрезе по типам

# In[60]:


comments_per_cloth = df.groupby('clothing_id')     .agg({'age':'count','class_name':'first'})     .reset_index()     .rename(columns = {'age':'count'})     .sort_values(by = 'count',ascending = False)

comments_per_cloth


# In[61]:


comments_per_cloth['count'].sum()/comments_per_cloth['clothing_id'].count()


# В среднем на одну вещь приходится по 19 комментариев

# ### Средний рейтинг по дивизионам

# Перед этим стоит пояснить, чем отличаются дивизионы одежды.
# <br> *General* - простая одежда средних размеров (вверх, низ, платья и т.п.)
# <br> *General Petite* - простая одежда (как General) только адаптированная под людей с низким ростом. Меньшие размеры и отдельные части одежды адаптированы под низких людей.
# <br> *Initmates* - домашняя одежда, нижнее белье

# Сделаем сводную таблицу со средними рейтингами по типам одежды и дивизионам

# In[63]:


division_class = df.groupby(['division_name','class_name'])['rating'].mean().reset_index()
#cгруппируем по дивизиону и типу одежды
division_class.pivot(index = 'class_name', columns = 'division_name', values = 'rating').fillna('-')


# Можем заметить, что General и General Petite сходятся по ассортименту во многих типах одежды, при этом имеют разные средние рейтинги. Уберем типы одежды, которые не повторяются в этих двух категориях и сравним средние. Нам необходимо исключить: casual bottoms, lounge, shorts.

# In[64]:


df[~df.class_name.isin(['Casual bottoms','Lounge','Shorts'])]     .groupby('division_name')['rating']     .mean().to_frame()
#выбрали данные без неповторяющихся типов одежды


# Средний рейтинг в General Petite чуть больше, чем в General. 
# Можно выдвинуть гипотезу: 
# Средний рейтинг одежды General Petite (то есть для более низких людей) больше, чем General (обычных средних размеров)

# In[ ]:




