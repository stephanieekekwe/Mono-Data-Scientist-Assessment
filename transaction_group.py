
import pandas as pd
import numpy as np


def group_transaction(transactions_data):
    transactions_data['date_transform'] = pd.to_datetime(
        transactions_data['date'])
    groups_by_days = pd.DataFrame(transactions_data.groupby("group")["date_transform"].apply(
        lambda x: (pd.Timedelta(abs(x.diff()).mean())).round(freq='D')))

    avg_day = []
    for i in groups_by_days['date_transform']:
        i = str(i)
        avg_day.append(i[:-13:])

    groups_by_days.drop('date_transform', inplace=True, axis=1)
    groups_by_days['avg_num_days'] = avg_day
    transactions_groups = pd.DataFrame(
        transactions_data.groupby('group')['_id'].unique())
    transactions_groups['avg_num_days'] = avg_day
    transactions_groups.reset_index(inplace=True)

    def collate_details(val):
        arr = transactions_data.loc[transactions_data['_id'] == val]
        typ = ''.join(arr['type'].values)
        narr = ''.join(arr['narration'].values)
        date = ''.join(arr['date'].values)
        amount = ''.join(str(x) for x in arr['amount'])
        return {"narration": narr,  "amount": amount, "type": typ, "date": date}

    data = dict()

    for i in transactions_groups['group']:
        data['group'+str(i)] = {'average_number_of_days_between_transactions': transactions_groups['avg_num_days']
                                [i],  "transactions": [collate_details(i) for i in transactions_groups['_id'][i]]}

    return data
