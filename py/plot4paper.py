#----------base-------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.dates import DateFormatter, num2date
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as ticker
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator

def mjd2y(mjd):
    y = 2000 + ((mjd - 51544)/365.25)
    return y
def mjd2d(mjd):
    # MJD 的起始日期是 1858-11-17 00:00:00 UTC
    base_date = pd.Timestamp('1858-11-17')
    delta = pd.to_timedelta(mjd, unit='D')
    dates = base_date + delta
    date = np.array(dates.dt.strftime('%y-%m-%d'))
    return date

#----------data-----------
uvotmmjd = np.array([57385.4, 58424.66, 58493.35, 58764.9, 59469.73, 60671.4, 54404.6, 54439.2, 54765.6, 54779.06, 55145.06, 55160.5, 55226.2, 55233.15, 55501.75, 55829, 
                    58145.12, 58640.76, 58496.39, 58502.04, 58770.1, 58770.76, 58789.59, 58849.25, 59482.59, 59519.39, 60650.14, 60636.26, 60327.38, 60313.83, 58452.45, 
                    56233.65, 56283.9, 56593.27, 56621.69, 56949.04, 56989.38, 57005.49, 57297.16, 57322.3, 57350.5, 57365.3, 57681.1, 57713.46, 57769.6, 57773.36, 58086.9, 57737.84
                    ])
uvotw2 = np.array([19.81, 20.11, 20.11, 19.60, 19.82, 19.94, 19.57, 19.61, 19.48, 19.62, 19.67, 19.81, 19.72, 19.81, 19.77, 19.76, 
                  19.84, 20.08, 19.94, 19.98, 19.95, 20.01, 19.94, 19.99, 19.69, 19.79, 20.11, 20.20, 20.08, 19.87, 20.04, 
                  19.68, 19.82, 19.90, 19.84, 19.73, 19.67, 19.76, 19.71, 19.78, 19.60, 19.76, 19.86, 19.87, 19.76, 19.97, 19.66, 19.73])
uvotw2err = np.array([0.07, 0.12, 0.13, 0.10, 0.16, 0.25, 0.17, 0.14, 0.11, 0.09, 0.07, 0.10, 0.06, 0.06, 0.05, 0.06, 
                     0.09, 0.14, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12, 0.11, 0.16, 0.20, 0.20, 0.20, 0.14, 0.18, 
                     0.05, 0.11, 0.06, 0.06, 0.10, 0.06, 0.06, 0.06, 0.06, 0.08, 0.07, 0.06, 0.06, 0.06, 0.09, 0.06, 0.07])

daysx = np.array([51903.721435,53738.798611,54291.714826,54993.318889,54995.313113,57386.405741,58310.090417,58844.768565,59385.262894,59387.268484])

daysupl = np.array([48437.8, 58491.533206,59377.268484,59379.268484,59381.275382])#首个为ROSAT数据

lflux_022 = 10**np.array([-12.4818, -12.6029, -12.7477, -12.4363, -12.3577, -13.47, -13.50, -13.71, -13.74, -13.49])
lf022_min = 10**(-np.array([0.03, 0.02, 0.04, 0.02, 0.03, 0.15, 0.45, 0.22, 0.17, 0.13])+np.array([-12.4818, -12.6029, -12.7477, -12.4363, -12.3577, -13.47, -13.50, -13.71, -13.74, -13.49]))
lf022_max = 10**(np.array([0.03, 0.02, 0.04, 0.02, 0.03, 0.14, 0.43, 0.16, 0.15, 0.11])+np.array([-12.4818, -12.6029, -12.7477, -12.4363, -12.3577, -13.47, -13.50, -13.71, -13.74, -13.49]))

flux_upl = np.array([4.332e-13, 4.085e-15, 2.083e-15, 3.604e-15, 2.643e-15])








#-------UV&X------------------
fig, ax1 = plt.subplots(figsize=(8, 4))
#ax1.errorbar(uvotmjd, uvotv,  yerr=[uvotverr,uvotverr], fmt='o', markersize=3, alpha=1, capsize=0, color='#440044', label='UVOT-v')
#ax1.errorbar(mjdcss, cssm, yerr=[cssmerr, cssmerr], fmt='o', markersize=3, alpha=1, capsize=0, color='y', label='CRTS-v')
uvw2_plot = ax1.errorbar(mjd2y(uvotmmjd), uvotw2,  yerr=[uvotw2err,uvotw2err], fmt='o', markersize=4, alpha=0.5, capsize=0, color='#884400', label='UVOT-UVw2')
ax1.set_ylabel('Magnitude', fontsize=p4pfs, color='black')
ax2 = ax1.twinx()
errorbar_plot = ax2.errorbar(mjd2y(daysx), lflux_022, yerr=[lflux_022-lf022_min, lf022_max-lflux_022], fmt='o', markersize=4, color='black', capsize=2, label='XMM Data,0.2-2keV')
scatter_plot = ax2.scatter(mjd2y(daysupl[1:]), flux_upl[1:], marker='v', s=10, color='black')
ax2.set_yscale('log')
ax2.set_ylabel('$F_{0.2-2 \mathrm{keV}}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=p4pfs-1, color='black')
ax2.legend([(errorbar_plot, scatter_plot), uvw2_plot], ['XMM-Newton, 0.2-2 keV', 'UVOT-UVw2'],
           handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower left')
ax2.tick_params(axis='y', labelcolor='black', labelsize=p4pls)
ax1.tick_params(axis='y', labelcolor='black', labelsize=p4pls)
ax1.tick_params(axis='x', labelcolor='black', labelsize=p4pls)
ax1.set_ylim(20.25,19.4)
ax1.set_xlim(2007,2022)
ax1.set_xlabel('Year', fontsize=p4pfs)
fig.savefig('UV&X.pdf',bbox_inches='tight')
plt.show()



#------------c+p_fit-------------------
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) 
from astropy import units as u

# 定义分段函数（常数 + 幂律）
# 函数在连接点 xc 处是连续的
def piecewise_cutoff_powerlaw(x, xc, a, c, power_index=-9.0 / 16.0):
    """
    分段函数：前半段为constant，后半段为cut off pl
    
    参数:
    x -- 自变量数组
    xc -- 连接点
    a -- 振幅
    c -- 指数衰减因子
    """
    # 幂律指数固定为 -9/16
    #power_index = -9.0 / 16.0
    
    # 新的后半段函数
    def cutoff_powerlaw_func(x_val):
        return a * ((x_val-xc+1)**power_index) * np.exp(-c * (x_val-xc+1) ** (-power_index))  #以xc为x=1
    
    # 计算在连接点的常数值
    constant_value = cutoff_powerlaw_func(xc)
    
    # 使用 np.piecewise 构建分段函数
    return np.piecewise(x, 
                        [x <= xc, x > xc], 
                        [lambda x: constant_value,   # x <= xc 时的函数
                         cutoff_powerlaw_func])       # x > xc 时的函数

def piecewise_constant_powerlaw(x, xc, a, n):
    """
    一个分段函数，x <= xc 时为常数，x > xc 时为幂律。
    为了保证在xc处连续，常数值c被设定为 a * xc**n。
    
    参数:
    x -- 自变量数组
    xc -- 连接点 (需要拟合的参数)
    a -- 幂律函数的振幅 (需要拟合的参数)
    n -- 幂律函数的指数 (需要拟合的参数)
    """
    # 计算在连接点的常数值以保证连续性
    constant_value = a * (1**n)
    
    # 使用 np.piecewise 根据条件应用不同函数
    return np.piecewise(x, 
                        [x <= xc, x > xc], 
                        [lambda x: constant_value,   # x <= xc 时的函数
                         lambda x: a * ((x-xc+1)**n)])       # x > xc 时的函数,以xc为x=1



fitmjd = mjd2y(np.hstack([daysupl, daysx]))-2000
#fitl022 = np.array([1.2366e+44, 8.6174e+43, 5.5813e+43, 1.2602e+44, 1.5165e+44, 1.1419e+43, 4.3811e+42, 4.9254e+42, 4.6777e+42, 8.8244e+42])
d_L = cosmo.luminosity_distance(0.38144562).to(u.cm).value
fitf022 = np.hstack([flux_upl, lflux_022]) *4*np.pi*d_L**2
err_upper = (np.hstack([flux_upl, lf022_max])*4*np.pi*d_L**2 - fitf022)
err_lower = (fitf022 - np.hstack([flux_upl*0.9, lf022_min])*4*np.pi*d_L**2)
y_err = (err_upper + err_lower) / 2.0


fitmjd = mjd2y(daysx)-2000
d_L = cosmo.luminosity_distance(0.38144562).to(u.cm).value
fitf022 = lflux_022 *4*np.pi*d_L**2
err_upper = (lf022_max)*4*np.pi*d_L**2 - fitf022
err_lower = fitf022 - lf022_min*4*np.pi*d_L**2
y_err = (err_upper + err_lower) / 2.0






search_range_start = fitmjd[int(len(fitmjd) * 0.3)]
search_range_end = fitmjd[int(len(fitmjd) * 0.45)]
xc_candidates = np.linspace(search_range_start, search_range_end, 100)

best_ssr = np.inf
best_params = None
# 遍历所有可能的xc值
for xc_candidate in xc_candidates:
    # 定义只带 a 和 c 参数的临时函数
    def temp_func(x, a, c):
        return piecewise_cutoff_powerlaw(x, xc_candidate, a, c)
    try:
        # 对每一个固定的xc，拟合 a 和 c
        # !! 为新参数 c 提供初始猜测和边界 !!
        initial_guess_ac = [2.4e44, 0.5] 
        bounds_ac = ([1e20, 0.4], [np.inf, 0.6]) # c 必须为正，但可以先不设上限
        popt_ac, _ = curve_fit(temp_func, fitmjd, fitf022, p0=initial_guess_ac, bounds=bounds_ac)
        
        # 计算当前xc下的残差平方和 (SSR)
        y_pred = temp_func(fitmjd, *popt_ac)
        chi2 = np.sum(((fitf022 - y_pred) / y_err)**2)
        
        if chi2 < best_ssr:
            best_ssr = chi2
            # 新的参数组合: [xc, a, c]
            best_params = [xc_candidate, popt_ac[0], popt_ac[1], -9.0/16.0]
    
    except RuntimeError:
        continue

tbest_ssr = np.inf
tbest_params = None
for xc_candidate in xc_candidates:
    # 定义只带 a 和 c 参数的临时函数
    def ttemp_func(x, a, c):
        return piecewise_cutoff_powerlaw(x, xc_candidate, a, c, -5.0/12.0)
    try:
        # 对每一个固定的xc，拟合 a 和 c
        # !! 为新参数 c 提供初始猜测和边界 !!
        initial_guess_ac = [0.38e45, 0.98] 
        bounds_ac = ([1e20, 0.9], [np.inf, 100]) # c 必须为正，但可以先不设上限
        popt_ac, _ = curve_fit(ttemp_func, fitmjd, fitf022, p0=initial_guess_ac, bounds=bounds_ac)
        
        # 计算当前xc下的残差平方和 (SSR)
        y_pred = ttemp_func(fitmjd, *popt_ac)
        chi2 = np.sum(((fitf022 - y_pred) / y_err)**2)
        
        if chi2 < tbest_ssr:
            tbest_ssr = chi2
            tbest_params = [xc_candidate, popt_ac[0], popt_ac[1], -5.0/12.0]
    
    except RuntimeError:
        continue

ttbest_ssr = np.inf
ttbest_params = None
for xc_candidate in xc_candidates:
    # 定义只带 a 和 n 参数的临时函数
    def tttemp_func(x, a, n):
        return piecewise_constant_powerlaw(x, xc_candidate, a, n)
    try:
        # 对每一个固定的xc，拟合 a 和 n
        # !! 为新参数 c 提供初始猜测和边界 !!
        initial_guess_ac = [1.5e44, -1.68] 
        bounds_ac = ([1e20, -10], [np.inf, -1]) # 
        popt_ac, _ = curve_fit(tttemp_func, fitmjd, fitf022, p0=initial_guess_ac, bounds=bounds_ac)
        
        # 计算当前xc下的残差平方和 (SSR)
        y_pred = tttemp_func(fitmjd, *popt_ac)
        chi2 = np.sum(((fitf022 - y_pred) / y_err)**2)
        
        if chi2 < ttbest_ssr:
            ttbest_ssr = chi2
            ttbest_params = [xc_candidate, popt_ac[0], popt_ac[1]]
    
    except RuntimeError:
        continue

print("\n--- 完成 ---")
if best_params:
    print(f"找到的最佳连接点 (xc): {best_params[0]:.2f}")
    print(f"对应的最佳参数 (a): {best_params[1]:.2e}")
    print(f"对应的最佳参数 (c): {best_params[2]:.2e}")
    print(f"最小残差平方和 (SSR): {best_ssr:.2e}")
    print("\n")
    print(f"找到的最佳连接点 (xc): {tbest_params[0]:.2f}")
    print(f"对应的最佳参数 (a): {tbest_params[1]:.2e}")
    print(f"对应的最佳参数 (c): {tbest_params[2]:.2e}")
    print(f"最小残差平方和 (SSR): {tbest_ssr:.2e}")
    print("\n")
    print(f"找到的最佳连接点 (xc): {ttbest_params[0]:.2f}")
    print(f"对应的最佳参数 (a): {ttbest_params[1]:.2e}")
    print(f"对应的最佳参数 (n): {ttbest_params[2]:.2e}")# 注意这里是 n
    print(f"最小残差平方和 (SSR): {ttbest_ssr:.2e}")
    
    
    # 可视化结果
    plt.figure(figsize=(8, 6))
    
    # 绘制原始数据点
    plt.errorbar(mjd2y(daysx)-2000, lflux_022*4*np.pi*d_L**2,  yerr=[(lflux_022-lf022_min)*4*np.pi*d_L**2, (lf022_max-lflux_022)*4*np.pi*d_L**2], fmt='o', markersize=3, color='r', capsize=2, label='XMM Data,0.2-2keV')
    plt.errorbar(mjd2y(daysupl[0])-2000, 4.332e-13*4*np.pi*d_L**2,  yerr=1.796e-13*4*np.pi*d_L**2, fmt='s', markersize=3, color='lightcoral', capsize=2, label='ROSAT Data,0.2-2keV')
    plt.scatter(mjd2y(daysupl)[1:]-2000, flux_upl[1:]*4*np.pi*d_L**2, marker='v', color='r')
    
    # 绘制拟合曲线
    x_plot = np.linspace(min(fitmjd), max(fitmjd), 500)
    plt.plot(x_plot, piecewise_cutoff_powerlaw(x_plot, *best_params), 'b-.', label=r'Flux ∝ t$^{-\frac{9}{16}}$' f'e^-({best_params[2]:.2f}'r'*t^$\frac{9}{16}$)')
    plt.plot(x_plot, piecewise_cutoff_powerlaw(x_plot, *tbest_params), 'y-.', label=r'Flux ∝ t$^{-\frac{5}{12}}$' f'e^-({tbest_params[2]:.2f}'r'*t^$\frac{5}{12}$)')
    plt.plot(x_plot, piecewise_constant_powerlaw(x_plot, *ttbest_params), 'g-.', label=f'Flux ∝ t^{ttbest_params[2]:.2f}')
    
    # 标记连接点
    plt.axvline(x=xc_fit, color='green', linestyle='--', label=f't$_0$ = {ttbest_params[0]:.2f}')
    # 标记连接点
    plt.axvline(x=best_params[0], color='b', linestyle='--', label=f't$_0$ = {best_params[0]:.2f}')
    # 标记连接点
    plt.axvline(x=tbest_params[0], color='y', linestyle='--', label=f't$_0$ = {tbest_params[0]:.2f}')
    
    # 设置图表样式
    plt.title('con+copl', fontsize=p4pfs)
    plt.xlabel('Year - 2000', fontsize=p4pfs)
    #plt.ylabel('$F_{0.2-2 \mathrm{keV}}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=p4pfs)
    plt.ylabel('$Lumin_{0.2-2 \mathrm{keV}}$ (erg s$^{-1}$)', fontsize=p4pfs)
    plt.legend(fontsize=p4pfs-2)
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('c+p_lc_of_022.pdf',bbox_inches='tight')
    plt.show()
else:
    print("拟合失败。尝试更改初始猜测值和边界")





