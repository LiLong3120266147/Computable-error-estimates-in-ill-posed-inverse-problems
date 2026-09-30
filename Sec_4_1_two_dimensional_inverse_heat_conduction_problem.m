% 4.1：二维逆热传导问题的全局后验误差估计
% 本程序实现以下完整流程：
%   (1) 构造二维初始温度 z_true；
%   (2) 用热方程正演算子 A 生成后时刻数据 u=A*z_true；
%   (3) 加入给定相对噪声，得到 u_delta；
%   (4) 用二阶 Tikhonov 正则化和偏差原则计算 z_delta；
%   (5) 构造后验可行集合: Omega(z) <= R_delta, ||A*z-u_delta|| <= Delta_delta；
%   (6) 按五类必要条件搜索候选极值点；
%   (7) 得到全局后验误差估计 epsilon = sup ||z-z_delta||.
%   (8) 固定双放大系数对 (C_res, C_Omega)，分别控制残差约束和稳定化约束；
%   (9) 自适应双放大系数对 (C_res^ad, C_Omega^ad)，二者分别向上保留两位小数，可以不相同；
%  (10) 每种后验最坏候选解单独图窗显示；C 曲线仅保留 C_min、C_res、C_Omega 和固定 C。

clc;clear;
close all;

a = zeros(1,10);      % 近似解真实相对误差
b = zeros(1,10);      % 固定单常数 C 的后验估计
c = zeros(1,10);      % 可容许 C_min 的后验估计，即单常数自适应方案
d = zeros(1,10);      % 固定双系数对 (C_res,C_Omega) 的后验估计
e = zeros(1,10);      % 自适应双系数对 (C_res^ad,C_Omega^ad) 的后验估计

C_min_plot = zeros(1,10);
C_omega_plot = zeros(1,10);       % 真解进入稳定化约束所需 C_Omega
C_residual_plot = zeros(1,10);    % 真解进入残差约束所需 C_res

C_pair_residual_plot = zeros(1,10);
C_pair_omega_plot = zeros(1,10);

C_adapt_residual_plot = zeros(1,10);
C_adapt_omega_plot = zeros(1,10);

fixed_feasible_plot = false(1,10);
pair_feasible_plot = false(1,10);
adapt_pair_feasible_plot = false(1,10);

for i = 1:10
%% ========================================================================
%  0. 实验参数
% =========================================================================
par.N = 256;              % 每个方向的网格点数
par.L = 0.5;              % 计算区域 [-L,L] x [-L,L]
par.a = 0.2;              % 热方程中的扩散系数
par.tau = 0.05;           % 要恢复的较早时刻
par.t = 0.075;            % 观测数据所在的较晚时刻
par.delta = 0.01*i;       % 数据相对噪声水平
par.seed = 20260913;      % 随机种子，保证每次运行结果相同

par.C_discrepancy = 1.01; % 偏差原则中的常数 C_bar > 1
par.C_posterior = 1.08;   % 固定单常数后验可行集放大系数 C > 1

% 固定双放大系数对
par.C_pair_residual = 1.01;
par.C_pair_omega    = 1.08;
par.use_pair_C      = true;

% 自适应双放大系数对
par.use_adapt_pair_C = true;
par.C_adapt_min = 1.01;

% C 计算模式：'fixed' 只计算固定 C； 'min'   只计算包含真值的最小可容许 C；'both'  两种方案同时计算并比较
par.C_mode = 'both';

% 参数扫描设置
par.n_alpha = 50;
par.n_beta = 50;
par.alpha_min = 1.0e-5;
par.alpha_max = 1.0e5;
par.beta_min = 1.0e-12;
par.beta_max = 1.0;

% 候选点选取与局部精化设置
par.n_refine_step1 = 100;      % Step 1 二维参数精化种子数
par.n_refine_1d = 10;          % Step 2-4 一维参数精化种子数
par.seed_separation_2d = 0.08; % Step 1 参数种子的最小归一化距离
par.seed_separation_1d = 1;    % 一维种子的最小网格间隔

% 数值容差
par.activity_tol = 1.0e-6;    % 活动约束相对容差
par.ineq_tol = 1.0e-6;        % 非活动约束容差
par.denom_tol = 1.0e-12;      % 跳过接近奇异的参数点
par.unique_tol = 1.0e-6;      % 候选向量去重容差 
par.max_report_candidates = 5;% 打印距离最大前5个候选点

par.verbose = true;           % 打印每一步的候选点统计
         
%% ========================================================================
%  1. 空间网格、精确解以及热方程正演算子
% =========================================================================
N = par.N;
x = linspace(-par.L, par.L, N + 1);
x(end) = [];                       % FFT使用周期边界条件, 删除该点以保证周期边界
[X,Y] = meshgrid(x,x);
dx = 2*par.L/N;                    % 网格步长

% 构造一个中心热源加环形热源
rxy = sqrt(X.^2 + Y.^2);           % 径向对称函数, 每个网格点到原点的距离
z1 = 0.5*exp(-(rxy/0.1).^2); 

theta = 45*pi/180;                 % 倾斜角度
Xr = cos(theta)*X + sin(theta)*Y;  % 旋转后的横坐标
Yr = -sin(theta)*X + cos(theta)*Y; % 旋转后的纵坐标
rho = sqrt((Xr/0.3).^2 + (Yr/0.2).^2);              
z2 = 0.2*exp(-((rho-1)/0.4).^2);

z_true = z1 + z2;

% Fourier 波数
k1 = 2*pi*ifftshift((-floor(N/2):ceil(N/2)-1)/(2*par.L));
[KX,KY] = meshgrid(k1,k1);
k2 = KX.^2 + KY.^2;

% 从tau演化到t的Fourier乘子：A_hat(k) = exp(-a^2 (t-tau) |k|^2)
dt = par.t - par.tau;
Ahat = exp(-(par.a^2)*dt*k2);

% 二阶稳定化泛函 Omega(z)=||z||_2^2+||Delta z||_2^2, 在 Fourier 空间中, 其权重为 1+|k|^4
Rhat = 1 + k2.^2;

ztrue_hat = fft2(z_true);
u_true = real(ifft2(Ahat.*ztrue_hat));

%% ========================================================================
%  2. 添加相对噪声，得到 u_delta
% =========================================================================
rng(par.seed,'twister');
noise = randn(N,N);
noise = noise/grid_norm(noise);

noise = par.delta*grid_norm(u_true)*noise;
u_delta = u_true + noise;
udelta_hat = fft2(u_delta);

%% ========================================================================
%  3. 二阶 Tikhonov 正则化，并用偏差原则选择 alpha
% =========================================================================
target_residual = par.C_discrepancy*par.delta*grid_norm(u_delta);
alpha_reg = choose_tikhonov_alpha(Ahat,Rhat,udelta_hat,N,target_residual);

z_delta_hat = conj(Ahat).*udelta_hat ./ (abs(Ahat).^2 + alpha_reg*Rhat);
z_delta = real(ifft2(z_delta_hat));

reg_residual = spectral_norm(Ahat.*z_delta_hat-udelta_hat,N);
Omega_delta = spectral_omega(z_delta_hat,Rhat,N);

%% ========================================================================
%  4. 构造全局后验可行集合
% =========================================================================
% 后验集合： Z_delta={z: Omega(z)<=R_delta,||Az-u_delta||<=Delta_delta}
norm_delta = spectral_norm(z_delta_hat,N);
norm_true = spectral_norm(ztrue_hat,N);

true_error = spectral_norm(ztrue_hat-z_delta_hat,N);
true_Omega = spectral_omega(ztrue_hat,Rhat,N);
true_data_residual = spectral_norm(Ahat.*ztrue_hat-udelta_hat,N);

% 两个约束分别给出的最小 C：C_Omega 保证 Omega(z_true)<=C*Omega(z_delta)；
%          C_residual 保证 ||Az_true-u_delta||<=C*||Az_delta-u_delta||
C_Omega = true_Omega/max(Omega_delta,eps);
C_residual = true_data_residual/max(reg_residual,eps);

% 包含真值所需的理论最小 C
C_min_raw = max([1,C_Omega,C_residual]);

% C 保留两位小数并向上取整，例如 1.020 保持为 1.02，1.022 取为 1.03
C_min = ceil((C_min_raw-1.0e-12)*100)/100;

% 理论要求 C>1，因此最低取 1.01
C_min = max(1.01,C_min);

C_min_plot(i) = C_min;
C_omega_plot(i) = C_Omega;
C_residual_plot(i) = C_residual;

C_pair_residual_plot(i) = par.C_pair_residual;
C_pair_omega_plot(i) = par.C_pair_omega;

C_adapt_residual = ceil_two_decimals(max(par.C_adapt_min,C_residual));
C_adapt_omega    = ceil_two_decimals(max(par.C_adapt_min,C_Omega));

C_adapt_residual_plot(i) = C_adapt_residual;
C_adapt_omega_plot(i)    = C_adapt_omega;

%% ========================================================================
%  5. 调用 Algorithm 1，计算全局后验误差估计
% =========================================================================
epsilon_fixed = NaN;
epsilon_min = NaN;
epsilon_pair = NaN;
epsilon_adapt_pair = NaN;

fixed_is_feasible = false;
min_is_feasible = false;
pair_is_feasible = false;
adapt_pair_is_feasible = false;

if strcmpi(par.C_mode,'fixed') || strcmpi(par.C_mode,'both')
    R_delta_fixed = par.C_posterior*Omega_delta;
    Delta_delta_fixed = par.C_posterior*reg_residual;

    fixed_is_feasible = true_Omega <= R_delta_fixed*(1+1e-10) && ...
                        true_data_residual <= Delta_delta_fixed*(1+1e-10);

    alg_fixed = algorithm1_fourier(Ahat,Rhat,udelta_hat,z_delta_hat, ...
        R_delta_fixed,Delta_delta_fixed,N,par);

    epsilon_fixed = alg_fixed.epsilon;
    z_worst_hat_fixed = alg_fixed.best.zhat;
    z_worst_fixed = real(ifft2(z_worst_hat_fixed)); %#ok<NASGU>
end

if strcmpi(par.C_mode,'min') || strcmpi(par.C_mode,'both')
    R_delta_min = C_min*Omega_delta;
    Delta_delta_min = C_min*reg_residual;

    min_is_feasible = true_Omega <= R_delta_min*(1+1e-10) && ...
                      true_data_residual <= Delta_delta_min*(1+1e-10);

    alg_min = algorithm1_fourier(Ahat,Rhat,udelta_hat,z_delta_hat, ...
        R_delta_min,Delta_delta_min,N,par);

    epsilon_min = alg_min.epsilon;
    z_worst_hat_min = alg_min.best.zhat;
    z_worst_min = real(ifft2(z_worst_hat_min)); %#ok<NASGU>
end


% 固定双放大系数对方案：残差约束和稳定化约束分别采用给定常值 C_pair_residual 和 C_pair_omega
if par.use_pair_C
    R_delta_pair = par.C_pair_omega*Omega_delta;
    Delta_delta_pair = par.C_pair_residual*reg_residual;

    pair_is_feasible = true_Omega <= R_delta_pair*(1+1e-10) && ...
                       true_data_residual <= Delta_delta_pair*(1+1e-10);

    alg_pair = algorithm1_fourier(Ahat,Rhat,udelta_hat,z_delta_hat, ...
        R_delta_pair,Delta_delta_pair,N,par);

    epsilon_pair = alg_pair.epsilon;
    z_worst_hat_pair = alg_pair.best.zhat;
    z_worst_pair = real(ifft2(z_worst_hat_pair)); %#ok<NASGU>
end

% 自适应双放大系数对方案：两个约束分别使用各自所需的最小放大系数，并向上保留两位小数
if par.use_adapt_pair_C
    R_delta_adapt_pair = C_adapt_omega*Omega_delta;
    Delta_delta_adapt_pair = C_adapt_residual*reg_residual;

    adapt_pair_is_feasible = true_Omega <= R_delta_adapt_pair*(1+1e-10) && ...
                             true_data_residual <= Delta_delta_adapt_pair*(1+1e-10);

    alg_adapt_pair = algorithm1_fourier(Ahat,Rhat,udelta_hat,z_delta_hat, ...
        R_delta_adapt_pair,Delta_delta_adapt_pair,N,par);

    epsilon_adapt_pair = alg_adapt_pair.epsilon;
    z_worst_hat_adapt_pair = alg_adapt_pair.best.zhat;
    z_worst_adapt_pair = real(ifft2(z_worst_hat_adapt_pair)); %#ok<NASGU>
end

%% ========================================================================
%  6. 输出数值结果
% =========================================================================
fprintf('\n============================================================\n');
fprintf('Example 4.1：二维逆热传导的全局后验误差估计\n');
fprintf('============================================================\n');
fprintf('网格规模                  N x N = %d x %d\n',N,N);
fprintf('设定相对噪声              delta = %.2f\n',par.delta);
fprintf('Tikhonov 参数             alpha = %.4e\n',alpha_reg);
fprintf('偏差原则目标残差                 = %.4e\n',target_residual);
fprintf('正则化解数据残差                 = %.4e\n',reg_residual);
fprintf('Omega(z_delta)                  = %.4e\n',Omega_delta);
fprintf('Omega(z_true)                   = %.4e\n',true_Omega);
fprintf('Delta_true                      = %.4e\n',true_data_residual);
fprintf('C_Omega                         = %.6f\n',C_Omega);
fprintf('C_residual                      = %.6f\n',C_residual);
fprintf('理论最小 C                     = %.6f\n',C_min_raw);
fprintf('向上保留两位小数后的 C_min      = %.2f\n',C_min);
fprintf('||z_true-z_delta||/||z_true||   = %.4e\n',true_error/norm_true);

if strcmpi(par.C_mode,'fixed') || strcmpi(par.C_mode,'both')
    fprintf('\n-------------------- 固定 C 方案 --------------------\n');
    fprintf('固定 C                          = %.2f\n',par.C_posterior);
    fprintf('R_delta=C*Omega(z_delta)        = %.4e\n',R_delta_fixed);
    fprintf('Delta_delta=C*residual          = %.4e\n',Delta_delta_fixed);
    fprintf('精确解是否属于后验可行集         = %d\n',fixed_is_feasible);
    fprintf('epsilon/||z_true||              = %.4e\n',epsilon_fixed/norm_true);
    fprintf('估计量/真实误差                 = %.6f\n',epsilon_fixed/max(true_error,eps));
    fprintf('产生最大值的 Algorithm 1 步骤   = Step %d\n',alg_fixed.best.step);
    fprintf('该候选点 Omega/R_delta          = %.8f\n',alg_fixed.best.omega/R_delta_fixed);
    fprintf('该候选点 residual/Delta_delta   = %.8f\n',alg_fixed.best.residual/Delta_delta_fixed);
    fprintf('各步骤最大候选值 [M1,...,M0]    =\n');
    disp(alg_fixed.M);
    fprintf('各步骤最终候选点数 [N1,...,N0]  =\n');
    disp(alg_fixed.counts);
    fprintf('全部去重候选点总数              = %d\n',numel(alg_fixed.candidates));

    print_algorithm_diagnostics(alg_fixed);
    print_candidate_table(alg_fixed.candidates,R_delta_fixed,Delta_delta_fixed,par.max_report_candidates);

    if ~fixed_is_feasible
        warning(['固定 C 下精确解没有落入后验可行集合，因此该 epsilon 不一定能覆盖真实误差。', ...
                 '可改用最小可容许 C 方案，或适当增大 par.C_posterior。']);
    end
end


if par.use_pair_C
    fprintf('\n---------------- 固定双系数对方案 ----------------\n');
    fprintf('固定 C_residual                 = %.2f\n',par.C_pair_residual);
    fprintf('固定 C_Omega                    = %.2f\n',par.C_pair_omega);
    fprintf('R_delta=C_Omega*Omega(z_delta)  = %.4e\n',R_delta_pair);
    fprintf('Delta_delta=C_res*residual      = %.4e\n',Delta_delta_pair);
    fprintf('精确解是否属于后验可行集         = %d\n',pair_is_feasible);
    fprintf('epsilon/||z_true||              = %.4e\n',epsilon_pair/norm_true);
    fprintf('估计量/真实误差                 = %.6f\n',epsilon_pair/max(true_error,eps));
    fprintf('产生最大值的 Algorithm 1 步骤   = Step %d\n',alg_pair.best.step);
    fprintf('该候选点 Omega/R_delta          = %.8f\n',alg_pair.best.omega/R_delta_pair);
    fprintf('该候选点 residual/Delta_delta   = %.8f\n',alg_pair.best.residual/Delta_delta_pair);
    fprintf('各步骤最大候选值 [M1,...,M0]    =\n');
    disp(alg_pair.M);
    fprintf('各步骤最终候选点数 [N1,...,N0]  =\n');
    disp(alg_pair.counts);
    fprintf('全部去重候选点总数              = %d\n',numel(alg_pair.candidates));

    print_algorithm_diagnostics(alg_pair);
    print_candidate_table(alg_pair.candidates,R_delta_pair,Delta_delta_pair,par.max_report_candidates);

    if ~pair_is_feasible
        warning(['固定双系数对下精确解没有落入后验可行集合。', ...
                 '可增大 par.C_pair_residual 或 par.C_pair_omega。']);
    end
end

if strcmpi(par.C_mode,'min') || strcmpi(par.C_mode,'both')
    fprintf('\n---------------- 最小可容许 C 方案 ----------------\n');
    fprintf('最小可容许 C                    = %.2f\n',C_min);
    fprintf('R_delta=C_min*Omega(z_delta)    = %.4e\n',R_delta_min);
    fprintf('Delta_delta=C_min*residual      = %.4e\n',Delta_delta_min);
    fprintf('精确解是否属于后验可行集         = %d\n',min_is_feasible);
    fprintf('epsilon/||z_true||              = %.4e\n',epsilon_min/norm_true);
    fprintf('估计量/真实误差                 = %.6f\n',epsilon_min/max(true_error,eps));
    fprintf('产生最大值的 Algorithm 1 步骤   = Step %d\n',alg_min.best.step);
    fprintf('该候选点 Omega/R_delta          = %.8f\n',alg_min.best.omega/R_delta_min);
    fprintf('该候选点 residual/Delta_delta   = %.8f\n',alg_min.best.residual/Delta_delta_min);
    fprintf('各步骤最大候选值 [M1,...,M0]    =\n');
    disp(alg_min.M);
    fprintf('各步骤最终候选点数 [N1,...,N0]  =\n');
    disp(alg_min.counts);
    fprintf('全部去重候选点总数              = %d\n',numel(alg_min.candidates));

    print_algorithm_diagnostics(alg_min);
    print_candidate_table(alg_min.candidates,R_delta_min,Delta_delta_min,par.max_report_candidates);
end

fprintf('============================================================\n\n');


if par.use_adapt_pair_C
    fprintf('\n---------------- 自适应双系数对方案 ----------------\n');
    fprintf('自适应 C_residual              = %.2f\n',C_adapt_residual);
    fprintf('自适应 C_Omega                 = %.2f\n',C_adapt_omega);
    fprintf('R_delta=C_Omega^ad*Omega(z_delta) = %.4e\n',R_delta_adapt_pair);
    fprintf('Delta_delta=C_res^ad*residual     = %.4e\n',Delta_delta_adapt_pair);
    fprintf('精确解是否属于后验可行集         = %d\n',adapt_pair_is_feasible);
    fprintf('epsilon/||z_true||              = %.4e\n',epsilon_adapt_pair/norm_true);
    fprintf('估计量/真实误差                 = %.6f\n',epsilon_adapt_pair/max(true_error,eps));
    fprintf('产生最大值的 Algorithm 1 步骤   = Step %d\n',alg_adapt_pair.best.step);
    fprintf('该候选点 Omega/R_delta          = %.8f\n',alg_adapt_pair.best.omega/R_delta_adapt_pair);
    fprintf('该候选点 residual/Delta_delta   = %.8f\n',alg_adapt_pair.best.residual/Delta_delta_adapt_pair);
    fprintf('各步骤最大候选值 [M1,...,M0]    =\n');
    disp(alg_adapt_pair.M);
    fprintf('各步骤最终候选点数 [N1,...,N0]  =\n');
    disp(alg_adapt_pair.counts);
    fprintf('全部去重候选点总数              = %d\n',numel(alg_adapt_pair.candidates));

    print_algorithm_diagnostics(alg_adapt_pair);
    print_candidate_table(alg_adapt_pair.candidates,R_delta_adapt_pair,Delta_delta_adapt_pair,par.max_report_candidates);
end

a(i) = true_error/norm_true;
b(i) = epsilon_fixed/norm_true;
c(i) = epsilon_min/norm_true;
d(i) = epsilon_pair/norm_true;
e(i) = epsilon_adapt_pair/norm_true;

fixed_feasible_plot(i) = fixed_is_feasible;
pair_feasible_plot(i) = pair_is_feasible;
adapt_pair_feasible_plot(i) = adapt_pair_is_feasible;

end
%% 绘制不同噪声水平下的真实误差与后验误差估计
delta_plot = 0.01:0.01:0.10;

% 统一绘图颜色和标记
color_true  = [0.0000 0.4470 0.7410];     % 蓝色：真实相对误差
color_fixed = [0.8500 0.3250 0.0980];     % 橙色：固定单常数 C
color_pair  = [0.6350 0.0780 0.1840];     % 深红色：固定双系数对
color_adapt_pair = [0.0000 0.5000 0.5000];% 蓝绿色：自适应双系数对
color_min   = [0.4940 0.1840 0.5560];     % 紫色：最小可容许 C
color_omega = [0.3010 0.7450 0.9330];     % 青蓝色：C_Omega
color_res   = [0.4660 0.6740 0.1880];     % 绿色：C_res
color_bad   = [0.9290 0.6940 0.1250];     % 黄色：真解不在可行集

%%
figure('Color','w','Name','真实误差与不同 C 方案下的后验误差估计');

plot(delta_plot, a, 'Color',color_true,'Marker','^','LineStyle','-','LineWidth',1.5,'MarkerSize',6,'DisplayName','近似解相对误差');
hold on;

if strcmpi(par.C_mode,'fixed') || strcmpi(par.C_mode,'both')
    plot(delta_plot, b, 'Color',color_fixed,'Marker','o','LineStyle','-','LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName',sprintf('固定 C=%.2f 的后验估计',par.C_posterior));

    bad = ~fixed_feasible_plot;
    if any(bad)
        plot(delta_plot(bad),b(bad),'Color',color_bad,'Marker','x','LineStyle','none','LineWidth',2,'MarkerSize',9, ...
            'DisplayName','真解不在可行集合中');
    end
end

if par.use_pair_C
    plot(delta_plot, d, 'Color',color_pair,'Marker','p','LineStyle','-','LineWidth',1.5,'MarkerSize',7, ...
        'DisplayName',sprintf('固定系数对 (C_{res},C_{\\Omega})=(%.2f,%.2f) 的后验估计', ...
        par.C_pair_residual,par.C_pair_omega));

    bad_pair = ~pair_feasible_plot;
    if any(bad_pair)
        plot(delta_plot(bad_pair),d(bad_pair),'Color',color_pair,'Marker','x','LineStyle','none','LineWidth',2,'MarkerSize',9, ...
            'DisplayName','真解不在固定系数对可行集合中');
    end
end

if par.use_adapt_pair_C
    plot(delta_plot, e, 'Color',color_adapt_pair,'Marker','*','LineStyle','-','LineWidth',1.5,'MarkerSize',7, ...
        'DisplayName','自适应系数对 (C_{res}^{ad},C_{\Omega}^{ad}) 的后验估计');

    bad_adapt_pair = ~adapt_pair_feasible_plot;
    if any(bad_adapt_pair)
        plot(delta_plot(bad_adapt_pair),e(bad_adapt_pair),'Color',color_adapt_pair,'Marker','x','LineStyle','none','LineWidth',2,'MarkerSize',9, ...
            'DisplayName','真解不在自适应系数对可行集合中');
    end
end

if strcmpi(par.C_mode,'min') || strcmpi(par.C_mode,'both')
    plot(delta_plot, c,'Color',color_min,'Marker','s','LineStyle','-','LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName','可容许 C_{min} 的后验估计');
end

hold off;grid on;box on;

xlabel('相对噪声水平 \delta');
ylabel('相对误差');

legend('Location','northwest');

xlim([delta_plot(1), delta_plot(end)]);
xticks(delta_plot);

set(gca,'FontSize',11);

%%
figure('Color','w','Name','后验可行集合放大系数 C');

plot(delta_plot,C_omega_plot,'Color',color_omega,'Marker','d','LineStyle','-','LineWidth',1.4,'MarkerSize',6, ...
    'DisplayName','C_{\Omega}');
hold on;

plot(delta_plot,C_residual_plot,'Color',color_res,'Marker','v','LineStyle','-','LineWidth',1.4,'MarkerSize',6, ...
    'DisplayName','C_{res}');

plot(delta_plot,C_min_plot,'Color',color_min,'Marker','s','LineStyle','-','LineWidth',1.6,'MarkerSize',6, ...
    'DisplayName','C_{min}');

plot(delta_plot,par.C_posterior*ones(size(delta_plot)),'Color',color_fixed,'Marker','o','LineStyle','-','LineWidth',1.5,'MarkerSize',6, ...
    'DisplayName',sprintf('固定 C=%.2f',par.C_posterior));

hold off; grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('可行集合放大系数');
legend('Location','northwest','Interpreter','tex');
xlim([delta_plot(1), delta_plot(end)]);
xticks(delta_plot);
set(gca,'FontSize',11);


%% ========================================================================
%  7. 绘图
% =========================================================================
% 单独重算一个指定噪声水平 delta_show，仅用于绘制真解、观测、近似解以及不同方案对应的极值解
delta_show = 0.05;

par_show = par;
par_show.delta = delta_show;
par_show.verbose = false;

% 重新构造该噪声水平下的数据
rng(par_show.seed,'twister');
noise_show = randn(N,N);
noise_show = noise_show/grid_norm(noise_show);
noise_show = par_show.delta*grid_norm(u_true)*noise_show;

u_delta_show = u_true + noise_show;
udelta_show_hat = fft2(u_delta_show);

% 重新计算该噪声水平下的 Tikhonov 近似解
target_residual_show = par_show.C_discrepancy*par_show.delta*grid_norm(u_delta_show);

alpha_show = choose_tikhonov_alpha(Ahat,Rhat,udelta_show_hat,N,target_residual_show);

z_delta_show_hat = conj(Ahat).*udelta_show_hat./(abs(Ahat).^2 + alpha_show*Rhat);

z_delta_show = real(ifft2(z_delta_show_hat));

reg_residual_show = spectral_norm(Ahat.*z_delta_show_hat-udelta_show_hat,N);

Omega_delta_show = spectral_omega(z_delta_show_hat,Rhat,N);

% 计算该噪声水平下最小可容许 C
true_Omega_show = spectral_omega(ztrue_hat,Rhat,N);

true_data_residual_show = spectral_norm(Ahat.*ztrue_hat-udelta_show_hat,N);

C_Omega_show = true_Omega_show/max(Omega_delta_show,eps);

C_residual_show = true_data_residual_show/max(reg_residual_show,eps);

C_min_raw_show = max([1,C_Omega_show,C_residual_show]);

C_min_show = ceil((C_min_raw_show-1.0e-12)*100)/100;

C_min_show = max(1.01,C_min_show);

C_adapt_residual_show = ceil_two_decimals(max(par_show.C_adapt_min,C_residual_show));
C_adapt_omega_show    = ceil_two_decimals(max(par_show.C_adapt_min,C_Omega_show));

% 固定 C 方案对应的后验最坏候选解
R_delta_fixed_show = par_show.C_posterior*Omega_delta_show;

Delta_delta_fixed_show = par_show.C_posterior*reg_residual_show;

alg_fixed_show = algorithm1_fourier(Ahat,Rhat,udelta_show_hat,z_delta_show_hat, ...
    R_delta_fixed_show,Delta_delta_fixed_show,N,par_show);

z_worst_fixed_show = real(ifft2(alg_fixed_show.best.zhat));


% 固定双系数对方案对应的后验最坏候选解
R_delta_pair_show = par_show.C_pair_omega*Omega_delta_show;

Delta_delta_pair_show = par_show.C_pair_residual*reg_residual_show;

alg_pair_show = algorithm1_fourier(Ahat,Rhat,udelta_show_hat,z_delta_show_hat, ...
    R_delta_pair_show,Delta_delta_pair_show,N,par_show);

z_worst_pair_show = real(ifft2(alg_pair_show.best.zhat));

% 自适应双系数对方案对应的后验最坏候选解
R_delta_adapt_pair_show = C_adapt_omega_show*Omega_delta_show;

Delta_delta_adapt_pair_show = C_adapt_residual_show*reg_residual_show;

alg_adapt_pair_show = algorithm1_fourier(Ahat,Rhat,udelta_show_hat,z_delta_show_hat, ...
    R_delta_adapt_pair_show,Delta_delta_adapt_pair_show,N,par_show);

z_worst_adapt_pair_show = real(ifft2(alg_adapt_pair_show.best.zhat));

% 最小可容许 C 方案对应的后验最坏候选解
R_delta_min_show = C_min_show*Omega_delta_show;

Delta_delta_min_show = C_min_show*reg_residual_show;

alg_min_show = algorithm1_fourier(Ahat,Rhat,udelta_show_hat,z_delta_show_hat, ...
    R_delta_min_show,Delta_delta_min_show,N,par_show);

z_worst_min_show = real(ifft2(alg_min_show.best.zhat));

% 为真解、近似解和两种最坏候选解使用统一色标，便于直接比较
solution_min = min([z_true(:); z_delta_show(:); z_worst_fixed_show(:); ...
    z_worst_pair_show(:); z_worst_adapt_pair_show(:); z_worst_min_show(:)]);

solution_max = max([z_true(:); z_delta_show(:); z_worst_fixed_show(:); ...
    z_worst_pair_show(:); z_worst_adapt_pair_show(:); z_worst_min_show(:)]);

%% 图 4：delta_show 下精确解、观测数据与 Tikhonov 近似解
figure('Color','w','Name',sprintf('delta=%.2f 下精确解、观测数据与Tikhonov近似解',delta_show));

tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile; imagesc(x,x,z_true);
axis image xy; colorbar;
caxis([solution_min solution_max]);
title('精确解 z_{true}');
xlabel('x'); ylabel('y');

nexttile; imagesc(x,x,u_delta_show);
axis image xy; colorbar;
title(sprintf('含噪观测 u_{\\delta}, \\delta=%.2f',delta_show),'Interpreter','tex');
xlabel('x'); ylabel('y');

nexttile; imagesc(x,x,z_delta_show);
axis image xy; colorbar;
caxis([solution_min solution_max]);
title('Tikhonov 近似解 z_{\delta}');
xlabel('x'); ylabel('y');

%% 固定单常数 C 对应的后验最坏候选解
figure('Color','w','Name',sprintf('delta=%.2f 固定 C 后验最坏候选解',delta_show));
imagesc(x,x,z_worst_fixed_show);
axis image xy; colorbar;
caxis([solution_min solution_max]);
title(sprintf('固定 C=%.2f 的后验最坏候选解',par_show.C_posterior));
xlabel('x'); ylabel('y');

%% 固定双系数对对应的后验最坏候选解
figure('Color','w','Name',sprintf('delta=%.2f 固定系数对后验最坏候选解',delta_show));
imagesc(x,x,z_worst_pair_show);
axis image xy; colorbar;
caxis([solution_min solution_max]);
title(sprintf('固定系数对 (C_{res},C_{\\Omega})=(%.2f,%.2f) 的后验最坏候选解', ...
    par_show.C_pair_residual,par_show.C_pair_omega));
xlabel('x'); ylabel('y');

%% 自适应双系数对对应的后验最坏候选解
figure('Color','w','Name',sprintf('delta=%.2f 自适应系数对后验最坏候选解',delta_show));
imagesc(x,x,z_worst_adapt_pair_show);
axis image xy; colorbar;
caxis([solution_min solution_max]);
title(sprintf('自适应系数对 (C_{res}^{ad},C_{\\Omega}^{ad})=(%.2f,%.2f) 的后验最坏候选解', ...
    C_adapt_residual_show,C_adapt_omega_show));
xlabel('x'); ylabel('y');

%% 可容许 C_min 对应的后验最坏候选解
figure('Color','w','Name',sprintf('delta=%.2f 可容许 C_min 后验最坏候选解',delta_show));
imagesc(x,x,z_worst_min_show);
axis image xy; colorbar;
caxis([solution_min solution_max]);
title(sprintf('可容许 C_{min}=%.2f 的后验最坏候选解',C_min_show));
xlabel('x'); ylabel('y');

%% delta_show 下中心截面对比
figure('Color','w','Name', sprintf('delta=%.2f 下中心截面对比',delta_show));

mid = floor(N/2)+1;

plot(x,diag(z_true),'Color',color_bad,'LineStyle','-','LineWidth',1.8, ...
    'DisplayName','精确解');
hold on;

plot(x,diag(z_delta_show),'Color',color_true,'LineStyle','--','LineWidth',1.8, ...
    'DisplayName','Tikhonov 近似解');

plot(x,diag(z_worst_fixed_show),'Color',color_fixed,'LineStyle',':','LineWidth',1.8, ...
    'DisplayName',sprintf('固定 C 的后验解'));

plot(x,diag(z_worst_pair_show),'Color',color_pair,'LineStyle','--','LineWidth',1.8, ...
    'DisplayName',sprintf('固定系数对的后验解'));

plot(x,diag(z_worst_adapt_pair_show),'Color',color_adapt_pair,'LineStyle','-','LineWidth',1.8, ...
    'DisplayName',sprintf('自适应系数对的后验解'));

plot(x,diag(z_worst_min_show),'Color',color_min,'LineStyle','-.','LineWidth',1.8, ...
    'DisplayName',sprintf('可容许 C_{min} 的后验解'));

hold off; grid on; box on;

xlabel('x');
ylabel('y=x');
legend('Location','northwest');
set(gca,'FontSize',11);

%% delta_show 下无噪声观测与含噪观测对比
figure('Color','w','Name',sprintf('delta=%.2f 下观测数据对比',delta_show));

tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; imagesc(x,x,u_true);
axis image xy; colorbar;
title('无噪声观测 u_{true}');
xlabel('x'); ylabel('y');

nexttile; imagesc(x,x,u_delta_show);
axis image xy; colorbar;
title(sprintf('含噪观测 u_{\\delta}, \\delta=%.2f',delta_show),'Interpreter','tex');
xlabel('x'); ylabel('y');

%% 向上保留小数点后两位
function value = ceil_two_decimals(x)
    value = ceil((x-1.0e-12)*100)/100;
end

%% 利用偏差原则选择 Tikhonov 正则化参数
function alpha = choose_tikhonov_alpha(Ahat,Rhat,uhat,N,target)
% 用对数尺度二分法求解 ||A z_alpha-u_delta|| = target,
% 其中 z_alpha_hat=Ahat^* uhat/(|Ahat|^2+alpha Rhat).

    residual = @(a) tikh_residual(a,Ahat,Rhat,uhat,N);

    alo = 1e-12;
    ahi = 1e2;

    while residual(alo) > target && alo > 1e-100
        alo = alo/10;
    end
    while residual(ahi) < target && ahi < 1e100
        ahi = ahi*10;
    end

    if residual(alo) > target
        error('无法找到偏差原则左端点：请检查噪声、网格和模型参数。');
    end
    if residual(ahi) < target
        error('无法找到偏差原则右端点：请增大 alpha 搜索上界。');
    end

    for k = 1:50
        amid = sqrt(alo*ahi);   % 几何平均值
        if residual(amid) < target
            alo = amid;
        else
            ahi = amid;
        end
    end
    alpha = sqrt(alo*ahi);
end

function value = tikh_residual(alpha,Ahat,Rhat,uhat,N)
    zhat = conj(Ahat).*uhat ./ (abs(Ahat).^2 + alpha*Rhat);
    value = spectral_norm(Ahat.*zhat-uhat,N);
end

%% Algorithm 1 在线性 Fourier 模型中的实现
function alg = algorithm1_fourier(Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par)
% 按 Algorithm 1 的五类必要条件搜索候选点

    logA = linspace(log(par.alpha_min),log(par.alpha_max),par.n_alpha);
    logB = linspace(log(par.beta_min),log(par.beta_max),par.n_beta);

    allCandidates = repmat(empty_candidate(),0,1);
    M = zeros(1,5);
    counts = zeros(1,5);    % 记录Step 1–Step 5最终保留的候选点数量
    diagnostics = repmat(empty_step_diagnostic(),5,1);

    %% ------------------------------- Step 1 ------------------------------
    if par.verbose
        fprintf('Algorithm 1 - Step 1：两个约束同时活动...\n');
    end

    scores = inf(par.n_alpha,par.n_beta);
    coarseCandidates = cell(par.n_alpha,par.n_beta); % 保存生成的粗网格候选点；
    coarseAccepted = repmat(empty_candidate(),0,1);  % 保存通过活动约束检查的候选点

    diag1 = empty_step_diagnostic();
    diag1.step = 1;                                  % 第1步
    diag1.grid_total = par.n_alpha*par.n_beta;       % 记录粗网格参数组合总数

    for ia = 1:par.n_alpha
        alpha = exp(logA(ia));

        for ib = 1:par.n_beta
            beta = exp(logB(ib));
            c = make_candidate(1,alpha,beta,Ahat,Rhat,uhat,zeta_hat,N,par);

            coarseCandidates{ia,ib} = c;

            if ~c.valid
                continue;
            end

            diag1.grid_valid = diag1.grid_valid+1;     % 能成功生成有限的候选点
            scores(ia,ib) = equality_score(c,R,Delta);

            if accept_candidate_mode(c,R,Delta,par,'both')
                coarseAccepted(end+1,1) = c; %#ok<AGROW>
            end
        end
    end

    diag1.coarse_accepted = numel(coarseAccepted);

    seeds = select_spread_seeds_2d(scores,par.n_refine_step1,par.seed_separation_2d);
    diag1.selected_seeds = size(seeds,1);

    refinedAccepted = repmat(empty_candidate(),0,1);

    for kk = 1:size(seeds,1)
        ia = seeds(kk,1);
        ib = seeds(kk,2);
        p0 = [logA(ia),logB(ib)];

        obj = @(p) step1_objective(p,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par);

        p = fminsearch(obj,p0,optimset('Display','off','MaxIter',500, ...
            'MaxFunEvals',1500,'TolX',1.0e-10,'TolFun',1.0e-14));

        diag1.refine_attempted = diag1.refine_attempted+1;  % 记录尝试精化的次数

        if numel(p)~=2 || any(~isfinite(p)) || ...          % 检查有两个参数且为有限值
           p(1)<logA(1) || p(1)>logA(end) || p(2)<logB(1) || p(2)>logB(end)
            continue;
        end

        c = make_candidate(1,exp(p(1)),exp(p(2)),Ahat,Rhat,uhat,zeta_hat,N,par);

        if c.valid           % 统计精化后有效的候选点数
            diag1.refine_valid = diag1.refine_valid+1;
        end

        if accept_candidate_mode(c,R,Delta,par,'both')
            refinedAccepted(end+1,1) = c; %#ok<AGROW>
        end
    end

    diag1.refined_accepted = numel(refinedAccepted);  % 记录局部精化后通过两个活动约束的候选点个数

    % 合并和去重后 Step 1 最终保留的候选点集合
    stepCandidates = unique_candidates_fourier([coarseAccepted;refinedAccepted],par.unique_tol);

    diag1.raw_accepted = numel(coarseAccepted)+numel(refinedAccepted); % 去重前候选点个数
    diag1.unique_candidates = numel(stepCandidates);                   % 去重后候选点个数
    diagnostics(1) = diag1;

    counts(1) = numel(stepCandidates);
    [M(1),allCandidates] = append_step(stepCandidates,allCandidates);

    %% ------------------------------- Step 2 ------------------------------
    if par.verbose
        fprintf('Algorithm 1 - Step 2：退化的双活动约束...\n');
    end

    [stepCandidates,diag2] = search_one_parameter_complete(2,logB,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,'both');

    diagnostics(2) = diag2;
    counts(2) = numel(stepCandidates);
    [M(2),allCandidates] = append_step(stepCandidates,allCandidates);

    %% ------------------------------- Step 3 ------------------------------
    if par.verbose
        fprintf('Algorithm 1 - Step 3：仅数据残差约束活动...\n');
    end

    [stepCandidates,diag3] = search_one_parameter_complete(3,logA,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,'residual');

    diagnostics(3) = diag3;
    counts(3) = numel(stepCandidates);
    [M(3),allCandidates] = append_step(stepCandidates,allCandidates);

    %% ------------------------------- Step 4 ------------------------------
    if par.verbose
        fprintf('Algorithm 1 - Step 4：仅先验约束活动...\n');
    end

    [stepCandidates,diag4] = search_one_parameter_complete(4,logB,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,'omega');

    diagnostics(4) = diag4;
    counts(4) = numel(stepCandidates);
    [M(4),allCandidates] = append_step(stepCandidates,allCandidates);

    %% ------------------------------- Step 5 ------------------------------
    if par.verbose
        fprintf('Algorithm 1 - Step 5：检查退化驻点...\n');
    end

    diag5 = empty_step_diagnostic();
    diag5.step = 5;
    diag5.grid_total = 2;        % 检查两个特殊候选点 Reta=0 和 Delta=0

    stepCandidates = repmat(empty_candidate(),0,1);

    % 构造第一个特殊候选点(零解)
    c0 = candidate_from_zhat(5,NaN,NaN,zeros(size(zeta_hat)),Ahat,Rhat,uhat,zeta_hat,N);
    if c0.valid                  % 检查零解是否能正常计算
        diag5.grid_valid = diag5.grid_valid+1;
    end
    if feasible(c0,R,Delta,par)  % 检查零解是否属于后验可行集合
        stepCandidates(end+1,1) = c0; %#ok<AGROW>
    end

    zls_hat = zeros(size(zeta_hat));     % 初始化最小二乘候选点
    idx = abs(Ahat)>1.0e-13;
    zls_hat(idx) = uhat(idx)./Ahat(idx); % 构造截断最小二乘解

    % 构造第二个特殊候选点
    cls = candidate_from_zhat(5,NaN,NaN,zls_hat,Ahat,Rhat,uhat,zeta_hat,N);
    if cls.valid                 % 检查第二个点是否能正常计算
        diag5.grid_valid = diag5.grid_valid+1;
    end
    if feasible(cls,R,Delta,par) % 检查第二个点是否属于后验可行集合
        stepCandidates(end+1,1) = cls; %#ok<AGROW>
    end

    diag5.coarse_accepted = numel(stepCandidates);    % 去重前
    diag5.raw_accepted = numel(stepCandidates);

    stepCandidates = unique_candidates_fourier(stepCandidates,par.unique_tol);

    diag5.unique_candidates = numel(stepCandidates);  % 去重后
    diagnostics(5) = diag5;

    counts(5) = numel(stepCandidates);
    [M(5),allCandidates] = append_step(stepCandidates,allCandidates);

    %% ------------------------------- Step 6 ------------------------------
    allCandidates = unique_candidates_fourier(allCandidates,par.unique_tol);  % 全局去重

    if isempty(allCandidates)
        error(['Algorithm 1 没有找到合格候选点。请检查参数范围、', ...
               '活动约束容差以及候选点统计输出。']);
    end

    distances = [allCandidates.distance];   % 提取每个候选点的距离
    [epsilon,imax] = max(distances);        % 找到最大距离及其位置

    alg = struct();
    alg.epsilon = epsilon;
    alg.M = M;
    alg.counts = counts;             % 保存每一步最终候选点数量
    alg.best = allCandidates(imax);  % 保存最坏候选点
    alg.candidates = allCandidates;  % 保存全部去重后的候选点
    alg.diagnostics = diagnostics;

end

%% Step 1 的参数精化目标：同时使两个活动约束相等
function value = step1_objective(p,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par)

    if numel(p)~=2 || any(~isfinite(p)) || ...
       p(1)<log(par.alpha_min) || p(1)>log(par.alpha_max) || ...
       p(2)<log(par.beta_min) || p(2)>log(par.beta_max)

       value = 1.0e12;
       return;
    end

    c = make_candidate(1,exp(p(1)),exp(p(2)),Ahat,Rhat,uhat,zeta_hat,N,par);

    if ~c.valid
        value = 1.0e12;
    else
        value = equality_score(c,R,Delta);
    end
end

%% Step 2--4：完整的一维候选族搜索：单参数候选族先粗搜索，再用 fminbnd 精化
function [candidates,diagInfo] = search_one_parameter_complete(step,logGrid,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,mode)

    ng = numel(logGrid);
    score = inf(ng,1);
    boundaryFunction = nan(ng,1);                      % 保存活动约束函数，用于Step 3、4检测根
    coarseCandidates = repmat(empty_candidate(),ng,1); % 保存每个粗网格点的完整候选结构
    coarseAccepted = repmat(empty_candidate(),0,1);    % 保存粗网格上已经满足约束的候选点

    diagInfo = empty_step_diagnostic();                % 记录当前步骤的选点和精化统计
    diagInfo.step = step;
    diagInfo.grid_total = ng;

    for ii = 1:ng     % 粗参数网格搜索
        q = exp(logGrid(ii));
        if step==2 || step==4
            c = make_candidate(step,NaN,q,Ahat,Rhat,uhat,zeta_hat,N,par);
        else
            c = make_candidate(step,q,NaN,Ahat,Rhat,uhat,zeta_hat,N,par);
        end

        coarseCandidates(ii) = c;    % 保存当前参数对应的候选点

        if ~c.valid
            continue;
        end

        diagInfo.grid_valid = diagInfo.grid_valid+1;     % 统计能够正常生成候选点的参数数量
        score(ii) = one_parameter_score(c,R,Delta,mode); % 计算当前候选点的评分

        switch mode
            case 'residual'
                boundaryFunction(ii) = c.residual/Delta-1;
            case 'omega'
                boundaryFunction(ii) = c.omega/R-1;
            case 'both'
                boundaryFunction(ii) = NaN;
        end

        if accept_candidate_mode(c,R,Delta,par,mode)      % 粗网格上直接接受候选点
            coarseAccepted(end+1,1) = c; %#ok<AGROW>
        end
    end
 
    diagInfo.coarse_accepted = numel(coarseAccepted);     % 记录粗网格直接通过多少点
    refinedAccepted = repmat(empty_candidate(),0,1);      % 保存fminbnd局部精化后通过的点
    rootAccepted = repmat(empty_candidate(),0,1);         % 保存fzero求活动约束根后通过的点
 
    % Step 3、4：对所有符号变化区间逐段求活动约束根
    if strcmp(mode,'residual') || strcmp(mode,'omega')
        for ii = 1:ng-1    % 检查每一对相邻参数点，若任一端点无效，就跳过当前区间
            if ~isfinite(boundaryFunction(ii)) || ~isfinite(boundaryFunction(ii+1))
                continue;
            end

            % 只有出现严格符号变化时才调用fzero，否则跳过
            if boundaryFunction(ii)==0 || boundaryFunction(ii+1)==0 || ...
               boundaryFunction(ii)*boundaryFunction(ii+1)>0
                continue;
            end

            try          % 相邻参数区间中用fzero求活动约束的根
                pRoot = fzero(@(p) one_parameter_boundary_function(p,step,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,mode), ...
                    [logGrid(ii),logGrid(ii+1)],optimset('Display','off','TolX',1.0e-12));
            catch
                continue;
            end

            qRoot = exp(pRoot);
            if step==4    % 用根对应的参数重新生成候选点
                c = make_candidate(step,NaN,qRoot,Ahat,Rhat,uhat,zeta_hat,N,par);
            else
                c = make_candidate(step,qRoot,NaN,Ahat,Rhat,uhat,zeta_hat,N,par);
            end

            diagInfo.root_intervals = diagInfo.root_intervals+1;   % 记录成功处理符号变化区间个数

            if accept_candidate_mode(c,R,Delta,par,mode)
                rootAccepted(end+1,1) = c; %#ok<AGROW>
            end
        end
    end

    % 对评分较小且分散的一维网格点做局部极小精化
    seedIndices = select_spread_seeds_1d(score,par.n_refine_1d,par.seed_separation_1d);
    diagInfo.selected_seeds = numel(seedIndices);    % 记录选取精化初值个数

    for kk = 1:numel(seedIndices)      % 局部精化
        ii = seedIndices(kk);
        il = max(1,ii-1);
        ir = min(ng,ii+1);

        left = logGrid(il);
        right = logGrid(ir);

        if left==right
            p = logGrid(ii);
        else
            obj = @(p) one_parameter_objective(p,step,Ahat,Rhat,uhat,zeta_hat, ...
                R,Delta,N,par,mode,logGrid(1),logGrid(end));

            p = fminbnd(obj,left,right,optimset('Display','off','TolX',1.0e-12,'MaxIter',300));
        end

        diagInfo.refine_attempted = diagInfo.refine_attempted+1;

        q = exp(p);
        if step==2 || step==4
            c = make_candidate(step,NaN,q,Ahat,Rhat,uhat,zeta_hat,N,par);
        else
            c = make_candidate(step,q,NaN,Ahat,Rhat,uhat,zeta_hat,N,par);
        end

        if c.valid
            diagInfo.refine_valid = diagInfo.refine_valid+1;
        end

        if accept_candidate_mode(c,R,Delta,par,mode)
            refinedAccepted(end+1,1) = c; %#ok<AGROW>
        end
    end

    diagInfo.refined_accepted = numel(refinedAccepted);    % 记录局部精化通过数量
    diagInfo.root_accepted = numel(rootAccepted);          % 记录fzero求根通过数量
    diagInfo.raw_accepted = numel(coarseAccepted)+numel(refinedAccepted)+numel(rootAccepted);
                                                           % 记录去重前得到合格结果总数
    candidates = unique_candidates_fourier([coarseAccepted;rootAccepted;refinedAccepted],par.unique_tol);
                                                           % 记录去重后真正不同的候选点数量
    diagInfo.unique_candidates = numel(candidates);
end

%% 给定对数参数p，生成对应候选点，计算该候选点距离活动约束等式还有多远
function value = one_parameter_objective(p,step,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,mode,pmin,pmax)

    if ~isfinite(p) || p<pmin || p>pmax
        value = 1.0e12;
        return;
    end

    q = exp(p);
    if step==2 || step==4
        c = make_candidate(step,NaN,q,Ahat,Rhat,uhat,zeta_hat,N,par);
    else
        c = make_candidate(step,q,NaN,Ahat,Rhat,uhat,zeta_hat,N,par);
    end

    if ~c.valid
        value = 1.0e12;
    else
        value = one_parameter_score(c,R,Delta,mode);
    end
end

%% 给fzero使用的活动约束求根函数 给定参数p=logq，生成对应候选点，返回活动约束距离等号差多少
function value = one_parameter_boundary_function(p,step,Ahat,Rhat,uhat,zeta_hat,R,Delta,N,par,mode)

    q = exp(p);
    if step==4
        c = make_candidate(step,NaN,q,Ahat,Rhat,uhat,zeta_hat,N,par);
    else
        c = make_candidate(step,q,NaN,Ahat,Rhat,uhat,zeta_hat,N,par);
    end

    if ~c.valid
        error('参数点接近候选公式的奇异位置。');
    end

    switch mode
        case 'residual'
            value = c.residual/Delta-1;
        case 'omega'
            value = c.omega/R-1;
        otherwise
            error('该模式不适合符号变化求根。');
    end
end

%% 候选点验收函数，判断候选点是否满足当前分支所要求的约束条件
function accept = accept_candidate_mode(c,R,Delta,par,mode)

    accept = false;

    if ~c.valid
        return;
    end

    switch mode
        case 'both'
            accept = active_equal(c.omega,R,par.activity_tol) && ...
                active_equal(c.residual,Delta,par.activity_tol);

        case 'residual'
            accept = active_equal(c.residual,Delta,par.activity_tol) && ...
                c.omega <= R*(1+par.ineq_tol);

        case 'omega'
            accept = active_equal(c.omega,R,par.activity_tol) && ...
                c.residual <= Delta*(1+par.ineq_tol);
    end
end

%% 根据 Step 类型选择评分方式
function value = one_parameter_score(c,R,Delta,mode)

    switch mode
        case 'both'
            value = equality_score(c,R,Delta);
        case 'residual'
            value = log(max(c.residual/Delta,realmin))^2;
        case 'omega'
            value = log(max(c.omega/R,realmin))^2;
        otherwise
            error('未知的单参数评分方式。');
    end
end

%% 计算“两个约束同时活动”的联合评分
function value = equality_score(c,R,Delta)

    value = log(max(c.omega/R,realmin))^2 + log(max(c.residual/Delta,realmin))^2;
end

%% 从二维参数网格中，选择nseed个“评分较小但彼此不要太接近”的网格点，作为fminsearch的初始参数点
function seeds = select_spread_seeds_2d(scores,nseed,minSeparation)

    [~,order] = sort(scores(:),'ascend');
    [na,nb] = size(scores);

    seeds = zeros(0,2);
    selectedNormalized = zeros(0,2);

    for kk = 1:numel(order)
        [ia,ib] = ind2sub(size(scores),order(kk));

        if ~isfinite(scores(ia,ib))
            continue;
        end

        q = [ (ia-1)/max(na-1,1),(ib-1)/max(nb-1,1)];

        if isempty(selectedNormalized)
            accept = true;
        else
            distances = sqrt(sum((selectedNormalized-q).^2,2));
            accept = all(distances>=minSeparation);
        end

        if accept
            seeds(end+1,:) = [ia,ib]; %#ok<AGROW>
            selectedNormalized(end+1,:) = q; %#ok<AGROW>
        end

        if size(seeds,1)>=nseed
            break;
        end
    end
end

%% 从一维参数网格中，选择nseed个“评分较小但彼此不要太接近”的网格点，作为fminsearch的初始参数点
function indices = select_spread_seeds_1d(score,nseed,minGap)

    [~,order] = sort(score,'ascend');
    indices = zeros(0,1);

    for kk = 1:numel(order)
        ii = order(kk);

        if ~isfinite(score(ii))
            continue;
        end

        if isempty(indices) || all(abs(indices-ii)>=minGap)
            indices(end+1,1) = ii; %#ok<AGROW>
        end

        if numel(indices)>=nseed
            break;
        end
    end
end

%% 候选点去重函数
function uniqueList = unique_candidates_fourier(candidates,tol)

    uniqueList = repmat(empty_candidate(),0,1);

    for kk = 1:numel(candidates)
        c = candidates(kk);

        if ~c.valid || isempty(c.zhat)
            continue;
        end

        duplicate = false;

        for jj = 1:numel(uniqueList)
            old = uniqueList(jj);

            difference = spectral_norm(c.zhat-old.zhat,size(c.zhat,1));
            scale = max([ ...
                1, ...
                spectral_norm(c.zhat,size(c.zhat,1)), ...
                spectral_norm(old.zhat,size(old.zhat,1))]);

            if difference <= tol*scale
                duplicate = true;
                break;
            end
        end

        if ~duplicate
            uniqueList(end+1,1) = c; %#ok<AGROW>
        end
    end
end

%% 生成空结构体
function d = empty_step_diagnostic()

    d = struct('step',0,'grid_total',0,'grid_valid',0,'coarse_accepted',0, ...
        'selected_seeds',0,'refine_attempted',0,'refine_valid',0,'refined_accepted',0, ...
        'root_intervals',0,'root_accepted',0,'raw_accepted',0, 'unique_candidates',0);
end

%% 打印后验估计参数
function print_algorithm_diagnostics(alg)

    fprintf('\n候选点选取统计：\n');
    fprintf(['Step | 网格总数 | 有效网格 | 粗网格通过 | 精化种子 | ', ...
             '精化有效 | 精化通过 | 根区间 | 根通过 | 去重后\n']);
    fprintf(['----------------------------------------------------------------', ...
        '----------------\n']);

    for kk = 1:numel(alg.diagnostics)
        d = alg.diagnostics(kk);

        fprintf([' %2d  | %8d | %8d | %10d | %8d | ','%8d | %8d | %6d | %6d | %6d\n'], ...
                d.step,d.grid_total,d.grid_valid,d.coarse_accepted, ...
                d.selected_seeds,d.refine_valid,d.refined_accepted, ...
                d.root_intervals,d.root_accepted,d.unique_candidates);
    end

    fprintf('\n');
end

%% 打印候选点表格
function print_candidate_table(candidates,R,Delta,maxRows)

    if isempty(candidates)
        fprintf('没有候选点可显示。\n');
        return;
    end

    [~,order] = sort([candidates.distance],'descend');
    numberToPrint = min(maxRows,numel(order));

    fprintf('按距离从大到小排列的候选点（最多显示 %d 个）：\n',numberToPrint);
    fprintf('序号  Step       alpha         beta       Omega/R   residual/Delta    distance\n');
    fprintf('-------------------------------------------------------------------------------\n');

    for kk = 1:numberToPrint
        c = candidates(order(kk));

        fprintf('%4d  %4d  %12.4e  %12.4e  %10.6f  %14.6f  %12.4e\n', ...
            kk,c.step,c.alpha,c.beta,c.omega/R,c.residual/Delta,c.distance);
    end

    fprintf('\n');
end

%% 利用 Fourier 对角化公式直接生成候选点
function c = make_candidate(step,alpha,beta,Ahat,Rhat,uhat,zeta_hat,N,par)
    switch step
        case 1
            den = 1-alpha*abs(Ahat).^2-beta*Rhat;
            if min(abs(den(:))) < par.denom_tol
                c = empty_candidate();
                return;
            end
            zhat = (zeta_hat-alpha*conj(Ahat).*uhat)./den;

        case 2
            den = abs(Ahat).^2+beta*Rhat;
            if min(abs(den(:))) < par.denom_tol
                c = empty_candidate();
                return;
            end
            zhat = conj(Ahat).*uhat./den;

        case 3
            den = 1-alpha*abs(Ahat).^2;
            if min(abs(den(:))) < par.denom_tol
                c = empty_candidate();
                return;
            end
            zhat = (zeta_hat-alpha*conj(Ahat).*uhat)./den;

        case 4
            den = 1-beta*Rhat;
            if min(abs(den(:))) < par.denom_tol
                c = empty_candidate();
                return;
            end
            zhat = zeta_hat./den;

        otherwise
            error('make_candidate 只接受 Step 1--4。');
    end
    c = candidate_from_zhat(step,alpha,beta,zhat,Ahat,Rhat,uhat,zeta_hat,N);
end

function c = candidate_from_zhat(step,alpha,beta,zhat,Ahat,Rhat,uhat,zeta_hat,N)
    c = empty_candidate();
    if any(~isfinite(real(zhat(:)))) || any(~isfinite(imag(zhat(:))))
        return;
    end

    c.step = step;
    c.alpha = alpha;
    c.beta = beta;
    c.zhat = zhat;
    c.omega = spectral_omega(zhat,Rhat,N);
    c.residual = spectral_norm(Ahat.*zhat-uhat,N);
    c.distance = spectral_norm(zhat-zeta_hat,N);
    c.valid = isfinite(c.omega) && isfinite(c.residual) && isfinite(c.distance);
end

%% 检查可容许相对精度
function tf = active_equal(value,target,tol)
    tf = abs(value-target) <= tol*max(target,eps);
end

function tf = feasible(c,R,Delta,par)
    tf = c.valid && c.omega <= R*(1+par.ineq_tol) && c.residual <= Delta*(1+par.ineq_tol);
end

function [M,allCandidates] = append_step(stepCandidates,allCandidates)
    if isempty(stepCandidates)
        M = 0;
        return;
    end
    M = max([stepCandidates.distance]);
    allCandidates = [allCandidates;stepCandidates]; %#ok<AGROW>
end

function c = empty_candidate()
    c = struct('step',0,'alpha',NaN,'beta',NaN,'zhat',[], ...
               'omega',Inf,'residual',Inf,'distance',-Inf,'valid',false);
end

%% 离散 L2 均方根范数：sqrt(mean(|v|^2))
function value = grid_norm(v)
    value = sqrt(mean(abs(v(:)).^2));
end

%% 由 Parseval 等式计算 grid_norm(ifft2(vhat))
function value = spectral_norm(vhat,N)
    value = sqrt(sum(abs(vhat(:)).^2))/N^2;
end

%% Omega(z)=||z||^2+||Delta z||^2 的 Fourier 表示
function value = spectral_omega(zhat,Rhat,N)
    value = real(sum(Rhat(:).*abs(zhat(:)).^2))/N^4;
end
