
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import wget,os
from pyecharts import options as opts
from pyecharts.render import make_snapshot
'''from snapshot_phantomjs import snapshot'''
from snapshot_selenium import snapshot
from pyecharts.charts    import Map, Line, Bar, Scatter, Pie


sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 确保中文显示正常
plt.rcParams['axes.unicode_minus'] = False   # 确保负号显示正常


# In[332]:


#数据集来源 丁香园新冠数据集 https://github.com/BlankerL/DXY-COVID-19-Data/releases
if os.path.isfile('./data1.csv'):
    os.remove('./data1.csv')
wget.download('https://github.com/BlankerL/DXY-COVID-19-Data/releases/download/2020.10.09/DXYArea.csv','data1.csv')
df = pd.read_csv(r'data1.csv', delimiter=',',header=0)
df=df[166148:216148]
# df.info()
# 从原始数据中截取50000条数据
# df = pd.read_csv(r'data.csv', delimiter=',',header=0)
# df=df[15000:65000]
# df.info()


# In[333]:


#字段解释
# countryName：国家名
# countryEnglishName：国家英文名
# provinceName：省名(境外省名与国家名相同)
# province_confirmedCount：省确诊数
# province_curedCount：省治愈数
# province_deadCount：省死亡数
# updateTime：更新时间
# cityName：城市名称


# In[334]:


#https://github.com/jianxu305/nCov2019_analysis/blob/master/src/utils.py
#丁香园的数据为实时数据，对于中国的部分需要用一些工具将数据规整
def load_chinese_raw(df):
    raw = df
    
    # the original CSV column names are in camel case, change to lower_case convention
    rename_dict = {'updateTime': 'update_time',
                   'provinceName': 'province_name',
                   'cityName': 'city_name',
                   'province_confirmedCount': 'province_confirmed',
                   'province_suspectedCount': 'province_suspected',
                   'province_deadCount': 'province_dead',
                   'province_curedCount': 'province_cured',
                   'city_confirmedCount': 'city_confirmed',
                   'city_suspectedCount': 'city_suspected',
                   'city_deadCount': 'city_dead',
                   'city_curedCount': 'city_cured'
                  }
    data = raw.rename(columns=rename_dict)
    data['update_time'] = pd.to_datetime(data['update_time'])  # original type of update_time after read_csv is 'str'
    data['update_date'] = data['update_time'].dt.date    # add date for daily aggregation, if without to_datetime, it would be a dateInt object, difficult to use
    # display basic info
    # print('Last update: ', data['update_time'].max())
    # print('Data date range: ', data['update_date'].min(), 'to', data['update_date'].max())
    # print('Number of rows in raw data: ', data.shape[0])
    return data 

def rename_cities(snapshots):
    dup_frm = snapshots[np.logical_and(
        np.logical_not(snapshots['city_name'].isnull()), 
        snapshots['city_name'].str.contains('（'))]
    rename_dict = {}
    if len(dup_frm) >= 0:
        rename_dict = dict([(name, name.split('（')[0]) for name in dup_frm['city_name']])
    
    rename_dict['吐鲁番市'] = '吐鲁番'   # raw data has both 吐鲁番市 and 吐鲁番, should be combined
    rename_dict['武汉来京人员'] = '外地来京人员'
    rename_dict['不明地区'] = '待明确地区'
    
    rename_dict['待明确'] = '待明确地区'
    rename_dict['未明确地区'] = '待明确地区'
    rename_dict['未知'] = '待明确地区'
    rename_dict['未知地区'] = '待明确地区'
    
    snapshots['city_name'] = snapshots['city_name'].replace(rename_dict)  # write back
    return snapshots

def add_daily_new(df, group_keys=['province_name', 'city_name'], diff_cols=['cum_confirmed', 'cum_dead', 'cum_cured'], date_col='update_date'):
    cols = ['confirmed', 'dead', 'cured']
    daily_new = df.groupby(group_keys)[diff_cols].transform(lambda x: np.diff(np.hstack([0, x])))
    
    new_cols = []
    for col in diff_cols:
        if 'cum_' in col:
            new_cols.append(col.replace('cum', 'new'))
        else:
            new_cols.append('new_' + col)
         
    daily_new = daily_new.rename(columns=dict(zip(diff_cols, new_cols)))
    df = pd.concat([df, daily_new], axis=1, join='outer')
    first_data_date = df[date_col].min()
    df[new_cols] = df[new_cols].where(df[date_col] != pd.Timestamp(first_data_date), np.nan)
    return df

def aggDaily(df):
    '''Aggregate the frequent time series data into a daily frame, ie, one entry per (date, province, city)'''
    frm_list = []
    drop_cols = ['province_' + field for field in ['confirmed', 'suspected', 'cured', 'dead']]  # these can be computed later
    drop_cols += ['provinceEnglishName', 'cityEnglishName', 'province_zipCode']
    for key, frm in df.drop(columns=drop_cols).sort_values(['update_date']).groupby(['province_name', 'city_name', 'update_date']):
        frm_list.append(frm.sort_values(['update_time'])[-1:])    # take the latest row within (city, date)
    out = pd.concat(frm_list).sort_values(['update_date', 'province_name', 'city_name'])
    to_names = [field for field in ['confirmed', 'suspected', 'cured', 'dead']]
    out = out.rename(columns=dict([('city_' + d, 'cum_' + d) for d in to_names]))
    out = out.rename(columns={'city_zipCode': 'zip_code'})
    out = out.drop(columns=['cum_suspected'])   
    out = add_daily_new(out)  
    new_col_order = ['update_date', 'continentName', 'countryName', 'continentEnglishName', 'countryEnglishName', 
                     'province_name', 'city_name', 'zip_code', 'cum_confirmed', 
                     'cum_cured', 'cum_dead', 'new_confirmed', 'new_cured', 'new_dead', 'update_time']
    if len(new_col_order) != len(out.columns):
        raise ValueError("Some columns are dropped: ", set(out.columns).difference(new_col_order))
    out = out[new_col_order]
    return out

def load_chinese_data(df):
    ''' This includes some basic cleaning'''
    data = load_chinese_raw(df)
    return rename_cities(data)
#import utils

ndf=load_chinese_data(df)
ndf=aggDaily(ndf).copy()
#ndf.to_csv('covid.csv')'''
#ndf = pd.read_csv(r'covid.csv', delimiter=',',header=0)

ndf.head()


# In[337]:


# 由于是差异计算出的新增，所以前两条数据不准确,剔除后得到4月数据
ndf.drop(ndf[ndf['update_date']==pd.Timestamp('2020-03-30')].index,inplace=True)
ndf.drop(ndf[ndf['update_date']==pd.Timestamp('2020-03-31')].index,inplace=True)
ndf.head()


# In[338]:


ndf=ndf.replace('待明确地区',np.nan)
ndf=ndf.dropna(subset=['city_name'])
ndf.head()


# In[339]:


#截至2020-05-01世界各地数据
wdf=df[df['updateTime']=='2020-04-30 22:22:39']
wdf.head()


# In[340]:


#删除国家名称为空的行，绘制地图需要国家英文名
wdf=wdf.dropna(subset=['countryEnglishName'])
wdf.head()
wdf=wdf.replace('United States of America','United States')


# In[341]:
print()
print('processing : 1.svg')
#1 截至2020-05-01全球疫情分布图(echarts)(map)
c = Map(init_opts=opts.InitOpts(renderer='svg'))
c.add("确诊人数", [list(z) for z in zip(wdf['countryEnglishName'].to_list(),
                                     [int(i) for i in wdf['province_confirmedCount'].to_list()])], "world")
c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
c.set_global_opts(
        title_opts=opts.TitleOpts(title="截至2020-05-01全球疫情分布图"),
        visualmap_opts=opts.VisualMapOpts(max_=200000),
    )
#c.render_notebook()
make_snapshot(snapshot,c.render(),"1.svg")

# In[335]:
#2
print('processing : 2.svg')
s=ndf['new_confirmed'].groupby([ndf['province_name']]).sum()
c = Map(init_opts=opts.InitOpts(renderer='svg'))
c.add("确诊人数", [list(z) for z in zip(
    [i.replace('省','').replace('市','').replace('自治区','') for i in s.index.to_list()],
    [int(i) for i in s.to_list()])], "china")
c.set_global_opts(
        title_opts=opts.TitleOpts(title="国内2020年4月疫情分布图"),
        visualmap_opts=opts.VisualMapOpts(max_=1500, is_piecewise=True),
    )
#c.render_notebook()
make_snapshot(snapshot,c.render(),"2.svg")

print('processing : 3.svg')
#3 2020年4月国内疫情散点图(echarts)(scatter)
s=ndf['new_confirmed'].groupby([ndf['province_name']]).sum()
c = Scatter(init_opts=opts.InitOpts(renderer='svg'))
c.add_xaxis(s.index.to_list())
c.add_yaxis("确诊数量", [int(i) for i in s.to_list()])
c.set_global_opts(
        title_opts=opts.TitleOpts(title="2020年4月国内疫情散点图"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-70)),
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
    )
#c.render_notebook()
make_snapshot(snapshot, c.render(), "3.svg")

# In[341]:4
print('processing : 4.svg')
s=ndf['new_confirmed'].groupby([ndf['update_date']]).sum()
c = Scatter(init_opts=opts.InitOpts(renderer='svg'))
c.add_xaxis(s.index.to_list())
c.add_yaxis("确诊人数", [int(i) for i in s.to_list()])
c.set_global_opts(
        title_opts=opts.TitleOpts(title="2020年4月全国累计新增确诊人数散点图"),
        xaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        visualmap_opts=opts.VisualMapOpts(type_="size", max_=2000, min_=1),
    )
#c.render_notebook()
make_snapshot(snapshot,c.render(),"4.svg")

#5 河南省4月新增确诊人数和新增治愈人数对比柱状图(echarts)(bar)
print('processing : 5.svg')
hdf=ndf[ndf['province_name']=='河南省']
conf=hdf['new_confirmed'].groupby(hdf['city_name']).sum()
curd=hdf['new_cured'].groupby(hdf['city_name']).sum()
c = Bar(init_opts=opts.InitOpts(renderer='svg'))
c.add_xaxis(conf.index.to_list())
c.add_yaxis("new_confirmed", conf.to_list())
c.add_yaxis("new_cured", curd.to_list())
c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
c.set_global_opts(title_opts=opts.TitleOpts(title="河南省4月新增确诊和新增治愈人数对比"),
                 xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)))
#c.render_notebook()
make_snapshot(snapshot,c.render(),"5.svg")

#6
print('processing : 6.svg')
ndf.drop(ndf[ndf['new_dead']==1290].index,inplace=True)#异常值
s=ndf['new_dead'].groupby(ndf['update_date']).sum()
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.lineplot(data=ndf['new_dead'].groupby(ndf['update_date']).sum())
plt.xticks(rotation=45)
plt.title('死亡新增',fontsize=20)
plt.subplot(1,2,2)
sns.lineplot(data=ndf['new_confirmed'].groupby(ndf['update_date']).sum())
plt.xticks(rotation=45)
plt.title('确诊新增',fontsize=20)
plt.tight_layout()
plt.savefig('6.svg')
#7
print('processing : 7.svg')
hdf=ndf[ndf['province_name']=='河南省']
s=ndf[ndf['province_name']=='河南省']['new_confirmed'].groupby(hdf['city_name']).sum()
t=pd.DataFrame({'city':s.index.to_list(),'count':s.to_list()})
plt.figure(figsize=(15,5))
sns.barplot(data=t,x='city',y='count')
plt.title('2020年4月河南省累计新增确诊人数',fontsize=20)
plt.savefig('7.svg')
#8
print('processing : 8.svg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14,5))
s=wdf['province_confirmedCount'].groupby(df['countryEnglishName']).sum().sort_values(ascending=False)
sns.lineplot(data=s)
plt.xticks(rotation=90)
plt.title('2020年4月海外累计新增确诊人数折线图',fontsize=20)
plt.tight_layout()
plt.savefig('8.svg')

#9
print('processing : 9.svg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
s=wdf['province_curedCount'].groupby(df['countryEnglishName']).sum().sort_values(ascending=False)
s=s[:20]
plt.figure(figsize=(8,8))
plt.pie(s,labels=s.index)
plt.title('2020年4月海外累计新增治愈人数前20',fontsize=20)
plt.tight_layout()
plt.savefig('9.svg')

#10 2020年4月海外累计新增死亡人数前20饼图(echarts)(pie)
print('processing : 10.svg')
s=wdf['province_deadCount'].groupby(df['countryEnglishName']).sum().sort_values(ascending=False)[:20]
c = Pie(init_opts=opts.InitOpts(renderer='svg'))
c.add("",
    [list(z) for z in zip(s.index.to_list(), s.to_list())],
    radius=["30%", "75%"],
    center=["35%", "55%"],
    )
c.set_global_opts(
    title_opts=opts.TitleOpts(title="2020年4月海外累计新增死亡人数前20"),
    legend_opts=opts.LegendOpts(pos_right="10",pos_top="10", orient="vertical")
    )
#c.render_notebook()
make_snapshot(snapshot,c.render(),"10.svg")
if os.path.isfile('./data1.csv'):
    os.remove('./render.html')
