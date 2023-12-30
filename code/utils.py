import numpy as np

def read_element_groups(reg='cen'):
    element_groups = {
        'H': [f'1H_{reg}'],
        'He': [f'4He_{reg}'],
        'C': [f'12C_{reg}', f'13C_{reg}'],
        'N': [f'14N_{reg}'],
        'O': [f'16O_{reg}', f'17O_{reg}', f'18O_{reg}'],
        'Ne': [f'20Ne_{reg}', f'22Ne_{reg}'],
        #'Al': [f'26Al_{reg}']
    }
    return element_groups



# current_sum = np.zeros_like(tab['time'])
# for reg in comps_dic:
#     element_groups = ut.read_element_groups(reg)
#     comps_dic[reg]['pie'] = []
#     for group in element_groups.values():
#         group_sum = np.sum([np.array(tab[element][step]) for element in group], axis=0)
#         next_sum = current_sum + group_sum
#         current_sum = next_sum
#         comps_dic[reg]['pie'].append(np.sum(group_sum))

# pie1 = ax.pie(comps_dic['surf']['pie'], radius=1,
#             colors=colors, startangle=90)

# pie2 = ax.pie(comps_dic['cen']['pie'], colors=colors, startangle=90, radius=0.6, wedgeprops=dict(linewidth=0, edgecolor='w'))