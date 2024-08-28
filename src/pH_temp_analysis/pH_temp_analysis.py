import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
import colorsys
from collections import defaultdict

''' 
    Statistical analysis of pH values and temperatures for successful sequences across 10 generations of each representative sequence 
'''



def main():
    # data path
    root = '../../data'
    model_type = 'oa_dm_640M'    
    # directory path
    top_dir = os.path.join(root, model_type)
    # info_file = os.path.join(root,'ec_species/motif_dict.csv')    
    # info_df = pd.read_csv(info_file)
    info_df = load_motif_dict()
    # data for plot
    pH_values, temp_ranges = collect_data(top_dir, info_df)

    # plot directory
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok = False)


    ''' Plot for pH'''
    box_plot(pH_values, "pH Distribution for Different IDs (Box Plot)", 'pH_boxplot_id.png', 'pH', 'ID')
    scatter_plot(pH_values, 'pH Values for Different IDs', 'pH_scatter_id.jpg', 'pH', 'ID', 'Species')
    violin_plot(pH_values, 'pH Values for Species', 'pH_violin_species.jpg', 'pH', 'Species', 'Species')

    ''' Plot for Temperature''' 
    stacked_bar_chart( df=temp_ranges,x='Temperature Range', y='ID', 
                      title='Temperature Range Distribution Across IDs', 
                      plot_name='temp_stacked_bar_id.png',legend_name='Temperature Range' )
    stacked_bar_chart( df=temp_ranges,x='Temperature Range', y='Species', 
                      title='Temperature Range Distribution Across Species', 
                      plot_name='temp_stacked_bar_species.png',legend_name='Temperature Range' )
    violin_plot(temp_ranges, 'Temperature Range for Species', 'temp_violin_species.jpg', 'Species', 'Temperature Range', 'Species', 
                plot_type = 'temp',figsize=(200, 80))
    heatmap(df=temp_ranges,x='Temperature Range', y='ID', z='Count',
            title='Temperature Range Distribution Across IDs', 
            plot_name='temp_heatmap_id.png')
    
    
    
    # save as csv
    pH_values.to_csv(os.path.join(plot_dir, 'pH_values_total.csv'))
    temp_ranges.to_csv(os.path.join(plot_dir, 'temp_ranges_total.csv'))

def load_motif_dict():
    # Dictionary containing file paths
    splits = {
        '3.2.1.1': 'data/3.2.1.1-00000-of-00001.parquet',
        '3.4.21.14': 'data/3.4.21.14-00000-of-00001.parquet',
        '3.4.21.62': 'data/3.4.21.62-00000-of-00001.parquet',
        '3.2.1.78': 'data/3.2.1.78-00000-of-00001.parquet',
        '3.1.1.3': 'data/3.1.1.3-00000-of-00001.parquet',
        '3.2.1.4': 'data/3.2.1.4-00000-of-00001.parquet'
    }

    # List to store DataFrames
    dfs = []

    # Read each parquet file and append DataFrame to list
    for key, file_path in splits.items():
        df = pd.read_parquet("hf://datasets/lun610200/detergent-motif/" + file_path)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


def collect_data(top_dir, info_df):
    pH_values = []
    temp_ranges = []

    for pdb in os.listdir(top_dir):
        # data path
        success_dir = os.path.join(top_dir, pdb, 'success')
        pH_file = os.path.join(success_dir, 'pH/ephod_pred.csv')
        temp_file = os.path.join(success_dir, 'temperature/mean_output.tsv')
        # check if there's successful sequence
        if not os.path.exists(os.path.join(success_dir, 'successes.csv')):
            continue
        success_df = pd.read_csv(os.path.join(success_dir, 'successes.csv'))
        if len(success_df) == 0 or not os.path.exists(pH_file) or not os.path.exists(temp_file):
            continue
        # check if ID is in motif.csv
        if pdb not in info_df['Entry'].values:
            continue
        # pH, temperature information
        pH_df = pd.read_csv(pH_file)
        temp_df = pd.read_csv(temp_file, sep='\t')
        
        ''' Species'''
        species = info_df[info_df['Entry']==pdb]['Species'].values[0]     
                
        ''' pH value''' 
        pH_list = pH_df['pHopt'].tolist()
        for pH in pH_list:
            pH_values.append({'ID': pdb, 'Species': species, 'pH': pH})

        ''' Temperature '''
        temp_df['Temperature Range'] = temp_df[['left_hand_label', 'right_hand_label']].apply(
            lambda row: row['left_hand_label'] if row['left_hand_label'] == row['right_hand_label'] else 'clash', axis=1
        )  
        temp_list = temp_df['Temperature Range'].tolist()
        for temp_range in temp_list:
            temp_ranges.append({'ID': pdb, 'Species': species, 'Temperature Range': temp_range})        
    
    
    return  pd.DataFrame(pH_values), pd.DataFrame(temp_ranges)





def lighten_color(color, amount=0.5):  
    # --------------------- SOURCE: @IanHincks ---------------------
    try:
        c = mc.cnames[color]        
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



def box_plot(df, title, plot_name, x, y, sorted=True, plot_dir='plot', figsize=(12,25)):
    if sorted:
        df.sort_values(by=x, inplace=True)

    fig, ax = plt.subplots(figsize=figsize)                           
    sns.set_theme(font_scale=1)
    flierprops = dict(marker='o', markersize=3)
    sns.boxplot(x=x, y=y, data=df, saturation=1, flierprops=flierprops, ax=ax, hue=y)    
    ax.set_title(title)
    for i, patch in enumerate(ax.patches):
        # 讓 facecolor 更亮一些
        col = lighten_color(patch.get_facecolor(), 1.2)
        patch.set_edgecolor(col)    

        # Loop over the lines for this boxplot
        for j in range(i*6, i*6+6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            line.set_linewidth(0.5)   # ADDITIONAL ADJUSTMENTT

    # add line on pH=7
    ax.axvline(x=7, color='darkgrey', linestyle='--', linewidth=2, label='pH = 7')

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')
    plt.close()

def scatter_plot(df, title, plot_name, x, y, z, sorted=True, plot_dir='plot', height=100):

    # Ensure data is sorted by the minimum value of the second element in each item (descending)
    if sorted:
        df.sort_values(by=x, inplace=True)

    sns.set_theme(font_scale = 5)
    g = sns.JointGrid(data=df, x=x, y=y, hue=z, height=height)#, palette=color_palette)
    g.plot_marginals(sns.histplot, kde=False)
    g.plot_joint(sns.scatterplot, s=4000)
    g.figure.subplots_adjust(top=0.95)
    g.figure.suptitle(title, fontsize=100)    
    g.savefig(os.path.join(plot_dir, plot_name)) #, bbox_inches='tight')


def violin_plot(df, title, plot_name, x, y, z='',plot_type='pH', sorted=True, plot_dir='plot', figsize=(40,60)):
    plt.figure(figsize=figsize)    
    if sorted:
        df.sort_values(by=x, inplace=True)
    if plot_type == 'temp':
        sns.set_theme(font_scale=10)
        # df = df.loc[df.index.repeat(df['Count'])].reset_index(drop=True)    
        y_order = ['clash', '65<=', '[60-65)', '[55-60)', '[50-55)', '[40-45)', '[45-50)', '<40']    
        
        df[y] = pd.Categorical(df[y], categories=y_order, ordered=True)
        
        sns.violinplot(data=df, x=x, y=y, hue=z)   
        plt.title(title)       
        sns.set_theme(font_scale=10)
        plt.xticks(rotation=45,ha='right') 
        plt.tight_layout()
    else:
        sns.violinplot(data=df, x=x, y=y, hue=z)   
        plt.title(title)  
        sns.set_theme(font_scale=10) 
        plt.tight_layout()   
    
    plt.savefig(os.path.join(plot_dir, plot_name))


# def count_plot(data, title, plot_name, x_label, ):


def stacked_bar_chart(df, title, plot_name, x, y, sorted=True, plot_dir='plot', figsize=(20, 25), kind='barh',
                      legend_name='Legend',
                      ordered_columns = ['<40', '[40-45)', '[45-50)', '[50-55)', '[55-60)', '[60-65)', '65<=', 'clash']):
    sns.set_theme(font_scale=1)
    if sorted:
        df.sort_values(by=x, inplace=True)
    
    temp_df = df.copy()

    plt.figure(figsize=figsize)


    # Calculate the sum of counts per y
    ''' 
    unstack() 操作將多層索引的 Series 轉換為一個 DataFrame
    fill_value=0 確保了如果某些組合沒有數據，會用 0 來填充，而不是 NaN。
    '''
    pivot = temp_df.groupby([y, x]).size().unstack(fill_value=0) 
    '''
    div() 是除法操作
    axis=0 表示除法操作是沿著列的方向（縱向）進行的。
    '''
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100 # Convert to percentage

    # Reorder columns if needed
    pivot = pivot.reindex(columns=ordered_columns, fill_value=0)

    n_colors = len(ordered_columns)
    color_palette = sns.color_palette("tab20", n_colors)# sns.color_palette("tab20", len(ordered_columns))

    # kind = 'barh' or 'bar'
    pivot.plot(kind=kind, stacked=True, figsize=figsize, color=color_palette)        

    plt.gca().invert_yaxis()
    plt.title(title, fontsize=20)
    plt.ylabel(y, fontsize=20)
    plt.xlabel('Percentage(%)', fontsize=20)
    plt.yticks(fontsize=14)  # Reduce font size for y-axis labels if needed
    plt.xticks(fontsize=14)
    plt.legend(title=legend_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')
    plt.show()


def heatmap(df, title, plot_name, x, y, z, sorted=True, plot_dir='plot', figsize=(15, 25), kind='barh',
            legend_name='Legend',
            ordered_columns = ['<40', '[40-45)', '[45-50)', '[50-55)', '[55-60)', '[60-65)', '65<=', 'clash']):
    if sorted:
        df.sort_values(by=x, inplace=True)

    temp_df = df.copy()
    pivot = temp_df.groupby([y, x], observed=False).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=ordered_columns, fill_value=0)
    
    # Reorder the columns in heatmap_pivot according to the desired order
    # pivot = pivot[ordered_columns]

    color_palette = sns.light_palette("#a275ac", as_cmap=True)
    color_palette = sns.light_palette((20, 60, 50), input="husl")
    # color_palette = sns.color_palette("husl", as_cmap=True)
    # Create heatmap
    plt.figure(figsize=figsize)
    cmaps = ['Greys', 'Greens', 'Oranges', 'BuGn', 'GnBu', 'PuBu', 'PuBuGn', 'RdPu', 'PuBuGn', 'YlOrBr', 'Reds', 'Blues', 'Puples', 'BuPu', 'OrRd', 'PuRd', 'YlGn', 'YlGnBu', 'YlOrRd']
    sns.set_theme()
    ax = sns.heatmap(pivot, cmap=color_palette, annot=False, cbar=True) # , cbar_kws={'alpha': 0.8})
    # Adjust y-ticks font size
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)  # Increase font size for y-ticks
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)  # Optional: Adjust font size for x-ticks

    # plt.grid(visible=True)
    plt.title(title, fontsize=20)
    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)
    plt.savefig(os.path.join(plot_dir, plot_name))
    

if __name__ == '__main__':
    main()

    


