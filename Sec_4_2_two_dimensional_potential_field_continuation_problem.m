% 4.2(1)：二维位场延拓反问题

% 主要步骤：
%   1. 构造源函数 w = 20*chi_T1 + chi_T2;
%   2. 采用周期 Fourier 乘子，在 x_3 = zeta 上构造与 FFT 离散模型一致的精确位场 z;
%   3. 在 N=512 细网格上生成周期 Fourier 模型数据并加入噪声，再将含噪数据限制到 N=128 的反演网格上;
%   4. 使用频谱乘子 Ahat(omega) = exp(-(nu-zeta)*|omega|);
%   5. 使用偏差原则选取正则化参数，并计算二阶 Tikhonov 近似解;
%   6. 在 D1+D2 组合区域上定义局部平均线性泛函，并用算法2计算局部后验误差;
%   7. 后验可行集合放大系数同时比较四种选取方式：
%      固定单常数 C、可容许 C_min、固定双系数对 (C_res,C_Omega)、
%      以及自适应双系数对 (C_res^ad,C_Omega^ad);
%   8. 利用 Bayev 的 Lagrange 原理，在同一离散模型上计算两个局部泛函的有限维最优恢复误差;
%   9. 比较局部真实相对误差、局部后验误差估计和最优恢复误差。

clc;clear;
close all;

%% ========================================================================
%  0. 参数设置
% =========================================================================
par.zeta = 0.15;                     % 待恢复位场所在平面
par.nu = 0.25;                       % 观测数据所在平面
par.delta_list = 0.01:0.01:0.10;     % 相对噪声水平
par.seed = 1218;                     % 固定随机种子

par.roi_min = 0.0;    % 设置计算区域 [0,1]x[0,1]
par.roi_max = 1.0;

par.N_fine = 512;     % 细网格用于生成模型数据
par.N = 128;          % 粗网格用于求解反问题

par.C_discrepancy = 1.01;    % 偏差原则参数

par.C_posterior = 1.23;       % 固定单常数后验可行集合放大常数
par.C_mode = 'both';          % 'fixed'、'min' 或 'both'

% 固定双放大系数对：
par.use_pair_C = true;
par.C_pair_residual = 1.01;
par.C_pair_omega = 1.23;

% 自适应双放大系数对：
par.use_adapt_pair_C = true;
par.C_adapt_min = 1.01;

% 两个局部矩形区域，格式为 [xmin xmax ymin ymax]
par.region1 = [0.2,0.44,0.2,0.44];
par.region2 = [0.45,0.75,0.45,0.75];

% 将两个局部区域合并为 D1+D2，局部泛函取两个矩形并集上的平均值
par.local_region_mode = 'D1+D2';

% 选取一个噪声水平，用于绘制近似解、后验极值候选解及对角线截面
par.delta_show = 0.02;

% Bayev 最优恢复误差使用模型实验中已知的真解先验半径与实际信息误差半径
% 下面两个因子可取 >=1；默认 1 表示使用刚好包含真解的半径
par.opt_prior_factor = 1.00;
par.opt_error_factor = 1.00;
par.opt_logt_min = -40;
par.opt_logt_max = 40;
par.opt_grid_size = 241;
par.opt_tol_x = 1.0e-10;

% 沿该线画出精确解、Tikhonov 解和局部后验误差棒
par.line_start = [0.0,0.0];
par.line_end = [1.0,1.0];

par.n_curve_points = 501;              % 在对角线上取稠密点个数，绘制光滑曲线
par.posterior_s = (0.08:0.06:0.98).';  % 局部后验估计点位置

% 逐点支撑函数问题中使用的一维对偶参数搜索设置
par.dual_logt_min = -40;     % 搜索下界
par.dual_logt_max = 12;      % 搜索上界
par.dual_grid_size = 161;
par.dual_tol_x = 1.0e-10;    % 局部精化时，logt 的容差
par.dual_feas_tol = 1.0e-8;  % 满足约束时允许的容差

% 输出选项
par.verbose = true;
par.plot_source = false;     % 源区域
par.show_selected_points = true;
par.warn_if_true_not_feasible = false;

%% ========================================================================
%  1. 构造细网格和源函数 w
% =========================================================================
box_min = par.roi_min;         % 区域左右端点
box_max = par.roi_max;
box_length = box_max-box_min;  % 区域长度

Nf = par.N_fine;
N = par.N;

dxf = box_length/Nf;    % 细网格步长
dx = box_length/N;      % 粗网格步长

xf = box_min+(0:Nf-1)*dxf;  % 构造网格坐标，标准 FFT 周期网格
x = box_min+(0:N-1)*dx;

[Xf,Yf] = meshgrid(xf,xf);  % 二维细网格

T1 = (Xf-0.32).^2+(Yf-0.32).^2 < 0.0004;
T2 = (Xf-0.60).^2-(Xf-0.60).*(Yf-0.60)+(Yf-0.60).^2 < 0.01;

w_fine = 20*double(T1)+1*double(T2);

%% ========================================================================
%  2. 根据周期 Fourier 模型计算精确解 z
% =========================================================================
% 使精确解、正向算子、二阶 Tikhonov 正则化以及 Omega 的 FFT 离散使用同一周期 Fourier 模型
% Poisson 延拓在 Fourier 域中的乘子为 exp(-h*|omega|)，因此
%       z_true_hat = exp(-zeta*|omega|) * w_hat.

fprintf('正在构造与周期 Fourier 离散一致的精确位场...\n');

[k2_fine,kabs_fine] = fourier_frequencies_2d(Nf,box_length);
what_fine = fft2(w_fine);

ztrue_fine_hat = exp(-par.zeta*kabs_fine).*what_fine;
z_true_fine = real(ifft2(ztrue_fine_hat));

if mod(Nf,N)~=0
    error('N_fine 必须是 N 的整数倍。');
end

stride = Nf/N;  % 降采样步长
z_true = z_true_fine(1:stride:end,1:stride:end);  % 每隔 4 个细网格点取一个点
ztrue_hat = fft2(z_true);  % 粗网格精确解的二维离散 Fourier 变换

%% ========================================================================
%  3. 生成观测数据和正则化权重
% =========================================================================
Ahat_fine = exp(-(par.nu-par.zeta)*kabs_fine);                % 构造细网格前向算子的 Fourier 乘子
u_true_fine = real(ifft2(Ahat_fine.*ztrue_fine_hat));         % 由同一周期 Fourier 模型生成精确数据

[k2,kabs] = fourier_frequencies_2d(N,box_length);  % 构造粗网格 Fourier 频率
Ahat = exp(-(par.nu-par.zeta)*kabs);               % 构造粗网格前向算子的 Fourier 乘子

u_true = u_true_fine(1:stride:end,1:stride:end);   % 将精确数据限制到反演网格上

% 检查细网格生成后限制得到的数据与粗网格周期 Fourier 模型的一致性
u_true_coarse_model = real(ifft2(Ahat.*ztrue_hat));
periodic_model_mismatch = grid_norm(u_true-u_true_coarse_model)/max(grid_norm(u_true),eps);

fprintf('周期 Fourier 模型粗网格一致性相对误差 = %.3e\n',periodic_model_mismatch);
if periodic_model_mismatch>1.0e-6
    warning(['细网格限制后的精确数据与粗网格 Fourier 模型存在不可忽略的差异。', ...
        '此时 C_res、偏差原则和最优恢复误差会同时受到离散模型误差影响。']);
end

% Omega(z)=||z||_{W_2^2}^2=||z||_2^2+||Delta z||_2^2
Rhat = 1+k2.^2;    % 正则化权重

%% ========================================================================
%  4. 局部矩形区域与线性泛函
% =========================================================================
x_roi = x;            % 为绘图更换明确的变量名
z_true_roi = z_true;

[X,Y] = meshgrid(x,x);

% 分别构造 D1、D2 两个矩形，并将其并集 D1+D2 作为一个组合局部区域
b1 = par.region1;
b2 = par.region2;

mask1 = X>=b1(1) & X<=b1(2) & Y>=b1(3) & Y<=b1(4);
mask2 = X>=b2(1) & X<=b2(2) & Y>=b2(3) & Y<=b2(4);
mask = mask1 | mask2;

if ~any(mask(:))
    error('组合局部区域 D1+D2 中没有粗网格节点，请调整矩形范围。');
end

% D1+D2 上的区域平均泛函：对两个矩形并集中的所有网格点统一归一化
weights = double(mask);
weights = weights/sum(weights(:));

regions = struct();
regions(1).name = 'D_1+D_2';
regions(1).bounds1 = b1;
regions(1).bounds2 = b2;
regions(1).mask1 = mask1;
regions(1).mask2 = mask2;
regions(1).mask = mask;
regions(1).weights = weights;
regions(1).area = sum(mask(:))*dx^2;
regions(1).true_value = sum(weights(:).*z_true(:));

n_region = 1;

%% ========================================================================
%  5. 在 N=512 细网格上生成高斯噪声
% =========================================================================
rng(par.seed,'twister');
noise_direction_fine = randn(Nf,Nf);    % 标准正态随机矩阵
noise_direction_fine = noise_direction_fine/grid_norm(noise_direction_fine);
                                        % 随机矩阵归一化

% 先取细网格随机方向在粗网格上的限制, 再确定统一噪声幅值, 使反演网格上的相对噪声严格满足
%       ||u_delta-u_true|| / ||u_true|| = delta
noise_direction_coarse = noise_direction_fine(1:stride:end,1:stride:end);
noise_direction_coarse_norm = grid_norm(noise_direction_coarse);
if noise_direction_coarse_norm<=eps
    error('限制到粗网格后的随机噪声方向范数过小。');
end

n_delta = numel(par.delta_list);             % 相对噪声水平个数
results = repmat(empty_result(),n_delta,1);  % 预分配结构体数组

%% ========================================================================
%  6. 反演计算、区域局部后验误差估计与最优恢复误差
% =========================================================================
for id = 1:n_delta

    delta = par.delta_list(id);

    if par.verbose
        fprintf('\n============================================================\n');
        fprintf('例8.6：delta = %.3f\n',delta);
        fprintf('============================================================\n');
    end

    % ---------------------------------------------------------------------
    % 6.1 添加细网格噪声并限制到粗网格
    % ---------------------------------------------------------------------
    noise_scale = delta*grid_norm(u_true)/noise_direction_coarse_norm;
    noise_fine = noise_scale*noise_direction_fine;
    u_delta_fine = u_true_fine+noise_fine;

    u_delta = u_delta_fine(1:stride:end,1:stride:end);

    % 绘制 delta_show 对应的含噪观测数据
    if abs(delta-par.delta_show) < 1.0e-12
        figure('Color','w','Name',sprintf('delta=%.2f 含噪观测数据',delta));
    
        u_delta_show=u_delta;
        imagesc(x,x,u_delta);
        axis image xy;
        colorbar;
    
        xlabel('x_1');
        ylabel('x_2');
    
        title(sprintf('含噪观测数据 u^\\delta, \\delta=%.2f',delta),'Interpreter','tex');
    
        set(gca,'FontSize',11);
    end

    noise = u_delta-u_true;
    udelta_hat = fft2(u_delta);

    noise_bound = delta*grid_norm(u_true);
    actual_noise = grid_norm(noise)/max(grid_norm(u_true),eps);
    fine_actual_noise = grid_norm(noise_fine)/max(grid_norm(u_true_fine),eps);

    % ---------------------------------------------------------------------
    % 6.2 二阶 Tikhonov 近似解正则化与偏差原则
    % ---------------------------------------------------------------------
    target_residual = par.C_discrepancy*noise_bound;

    alpha_reg = choose_tikhonov_alpha(Ahat,Rhat,udelta_hat,N,target_residual);

    z_delta_hat = conj(Ahat).*udelta_hat./(abs(Ahat).^2+alpha_reg*Rhat);
    z_delta = real(ifft2(z_delta_hat));

    reg_residual = spectral_norm(Ahat.*z_delta_hat-udelta_hat,N);
    Omega_delta = spectral_omega(z_delta_hat,Rhat,N);

    % ---------------------------------------------------------------------
    % 6.3 构造两种后验可行集合放大系数 C
    % ---------------------------------------------------------------------
    true_Omega = spectral_omega(ztrue_hat,Rhat,N);
    true_residual = spectral_norm(Ahat.*ztrue_hat-udelta_hat,N);

    C_Omega = true_Omega/max(Omega_delta,eps);
    C_residual = true_residual/max(reg_residual,eps);

    C_min_raw = max([1,C_Omega,C_residual]);
    % 向上保留两位小数, 例如 1.022 -> 1.03
    C_min = ceil_to_two_decimals(C_min_raw);
    C_min = max(1.01,C_min);

    C_fixed = par.C_posterior;

    % 固定单常数 C：残差约束和稳定化约束使用同一个 C_fixed
    R_delta_fixed = C_fixed*Omega_delta;
    Delta_delta_fixed = C_fixed*reg_residual;
    fixed_is_feasible = true_Omega<=R_delta_fixed*(1+1.0e-10) && ...
                        true_residual<=Delta_delta_fixed*(1+1.0e-10);

    % 可容许单常数 C_min：取同时包含真解的单个最小放大系数
    R_delta_min = C_min*Omega_delta;
    Delta_delta_min = C_min*reg_residual;
    min_is_feasible = true_Omega<=R_delta_min*(1+1.0e-10) && ...
                      true_residual<=Delta_delta_min*(1+1.0e-10);

    % 固定双系数对：(C_res,C_Omega)
    C_pair_residual = par.C_pair_residual;
    C_pair_omega = par.C_pair_omega;
    R_delta_pair = C_pair_omega*Omega_delta;
    Delta_delta_pair = C_pair_residual*reg_residual;
    pair_is_feasible = true_Omega<=R_delta_pair*(1+1.0e-10) && ...
                       true_residual<=Delta_delta_pair*(1+1.0e-10);

    % 自适应双系数对：两个系数分别向上保留两位小数
    C_adapt_residual = ceil_to_two_decimals(max(par.C_adapt_min,C_residual));
    C_adapt_omega = ceil_to_two_decimals(max(par.C_adapt_min,C_Omega));
    R_delta_adapt_pair = C_adapt_omega*Omega_delta;
    Delta_delta_adapt_pair = C_adapt_residual*reg_residual;
    adapt_pair_is_feasible = true_Omega<=R_delta_adapt_pair*(1+1.0e-10) && ...
                             true_residual<=Delta_delta_adapt_pair*(1+1.0e-10);

    % ---------------------------------------------------------------------
    % 6.4 算法2：两个矩形区域上的局部后验误差
    % ---------------------------------------------------------------------
    local_fixed = [];
    local_min = [];
    local_pair = [];
    local_adapt_pair = [];

    if strcmpi(par.C_mode,'fixed') || strcmpi(par.C_mode,'both')
        local_fixed = algorithm2_region_fourier_fast(Ahat,Rhat,udelta_hat,z_delta_hat, ...
            R_delta_fixed,Delta_delta_fixed,regions,N,par);
    end

    if strcmpi(par.C_mode,'min') || strcmpi(par.C_mode,'both')
        local_min = algorithm2_region_fourier_fast(Ahat,Rhat,udelta_hat,z_delta_hat, ...
            R_delta_min,Delta_delta_min,regions,N,par);
    end

    if par.use_pair_C
        local_pair = algorithm2_region_fourier_fast(Ahat,Rhat,udelta_hat,z_delta_hat, ...
            R_delta_pair,Delta_delta_pair,regions,N,par);
    end

    if par.use_adapt_pair_C
        local_adapt_pair = algorithm2_region_fourier_fast(Ahat,Rhat,udelta_hat,z_delta_hat, ...
            R_delta_adapt_pair,Delta_delta_adapt_pair,regions,N,par);
    end

    % 当前近似解和真解在两个区域平均泛函上的取值
    functional_true = zeros(n_region,1);
    functional_delta = zeros(n_region,1);
    true_local_error = zeros(n_region,1);
    true_local_relative_error = zeros(n_region,1);

    for ir = 1:n_region
        wgt = regions(ir).weights;
        functional_true(ir) = regions(ir).true_value;
        functional_delta(ir) = sum(wgt(:).*z_delta(:));
        true_local_error(ir) = abs(functional_delta(ir)-functional_true(ir));
        true_local_relative_error(ir) = ...
            true_local_error(ir)/max(abs(functional_true(ir)),eps);
    end

    % 数值一致性检查: 若真解属于对应后验可行集合，则其局部泛函值必须位于 [fmin,fmax] 内
    check_tol = 1.0e-8;
    if ~isempty(local_fixed) && fixed_is_feasible
        for ir = 1:n_region
            scale_check = max([1,abs(functional_true(ir)), ...
                abs(local_fixed.fmin(ir)),abs(local_fixed.fmax(ir))]);
            tol_ir = check_tol*scale_check;

            if functional_true(ir)<local_fixed.fmin(ir)-tol_ir || ...
               functional_true(ir)>local_fixed.fmax(ir)+tol_ir
                warning('固定 C：%s 的真值泛函未落在 Algorithm 2 给出的区间内。', ...
                    regions(ir).name);
            end

            if local_fixed.E_functional(ir)+tol_ir<true_local_error(ir)
                warning('固定 C：%s 的局部后验误差未覆盖真实局部误差。', ...
                    regions(ir).name);
            end
        end
    end

    if ~isempty(local_min) && min_is_feasible
        for ir = 1:n_region
            scale_check = max([1,abs(functional_true(ir)), ...
                abs(local_min.fmin(ir)),abs(local_min.fmax(ir))]);
            tol_ir = check_tol*scale_check;

            if functional_true(ir)<local_min.fmin(ir)-tol_ir || ...
               functional_true(ir)>local_min.fmax(ir)+tol_ir
                warning('最小可容许 C：%s 的真值泛函未落在 Algorithm 2 给出的区间内。', ...
                    regions(ir).name);
            end

            if local_min.E_functional(ir)+tol_ir<true_local_error(ir)
                warning('最小可容许 C：%s 的局部后验误差未覆盖真实局部误差。', ...
                    regions(ir).name);
            end
        end
    end

    if ~isempty(local_pair) && pair_is_feasible
        for ir = 1:n_region
            scale_check = max([1,abs(functional_true(ir)), ...
                abs(local_pair.fmin(ir)),abs(local_pair.fmax(ir))]);
            tol_ir = check_tol*scale_check;

            if functional_true(ir)<local_pair.fmin(ir)-tol_ir || ...
               functional_true(ir)>local_pair.fmax(ir)+tol_ir
                warning('固定双系数对：%s 的真值泛函未落在 Algorithm 2 给出的区间内。', ...
                    regions(ir).name);
            end

            if local_pair.E_functional(ir)+tol_ir<true_local_error(ir)
                warning('固定双系数对：%s 的局部后验误差未覆盖真实局部误差。', ...
                    regions(ir).name);
            end
        end
    end

    if ~isempty(local_adapt_pair) && adapt_pair_is_feasible
        for ir = 1:n_region
            scale_check = max([1,abs(functional_true(ir)), ...
                abs(local_adapt_pair.fmin(ir)),abs(local_adapt_pair.fmax(ir))]);
            tol_ir = check_tol*scale_check;

            if functional_true(ir)<local_adapt_pair.fmin(ir)-tol_ir || ...
               functional_true(ir)>local_adapt_pair.fmax(ir)+tol_ir
                warning('自适应双系数对：%s 的真值泛函未落在 Algorithm 2 给出的区间内。', ...
                    regions(ir).name);
            end

            if local_adapt_pair.E_functional(ir)+tol_ir<true_local_error(ir)
                warning('自适应双系数对：%s 的局部后验误差未覆盖真实局部误差。', ...
                    regions(ir).name);
            end
        end
    end

    % ---------------------------------------------------------------------
    % 6.5 Bayev Lagrange 原理：两个局部泛函的有限维最优恢复误差
    % ---------------------------------------------------------------------
    % 最优恢复使用凸平衡先验集 M_R = {z : Omega(z) <= R_opt}
    % 和凸平衡误差邻域 O = {e : ||e|| <= Delta_opt}
    % 在模型实验中真解已知，因此取刚好包含真解的 R_opt 和 Delta_opt
    R_opt = par.opt_prior_factor*true_Omega;
    Delta_opt = par.opt_error_factor*true_residual;

    optimal_recovery = repmat(empty_optimal_recovery(),n_region,1);
    optimal_relative_error = zeros(n_region,1);

    for ir = 1:n_region
        optimal_recovery(ir) = optimal_recovery_error_fourier( ...
            Ahat,Rhat,regions(ir).weights,R_opt,Delta_opt,N,par);

        optimal_relative_error(ir) = ...
            optimal_recovery(ir).error/max(abs(functional_true(ir)),eps);
    end

    % ---------------------------------------------------------------------
    % 6.6 保存当前噪声水平下的计算结果
    % ---------------------------------------------------------------------
    true_L2_error = spectral_norm(ztrue_hat-z_delta_hat,N);
    norm_true = spectral_norm(ztrue_hat,N);

    result = empty_result();
    result.delta = delta;
    result.actual_noise = actual_noise;
    result.alpha = alpha_reg;
    result.target_residual = target_residual;
    result.reg_residual = reg_residual;
    result.Omega_delta = Omega_delta;
    result.true_Omega = true_Omega;
    result.true_residual = true_residual;
    result.true_relative_L2_error = true_L2_error/max(norm_true,eps);
    result.z_delta = z_delta;

    result.C_Omega = C_Omega;
    result.C_residual = C_residual;
    result.C_min_raw = C_min_raw;
    result.C_min = C_min;
    result.C_fixed = C_fixed;
    result.C_pair_residual = C_pair_residual;
    result.C_pair_omega = C_pair_omega;
    result.C_adapt_residual = C_adapt_residual;
    result.C_adapt_omega = C_adapt_omega;
    result.fixed_is_feasible = fixed_is_feasible;
    result.min_is_feasible = min_is_feasible;
    result.pair_is_feasible = pair_is_feasible;
    result.adapt_pair_is_feasible = adapt_pair_is_feasible;
    result.R_delta_fixed = R_delta_fixed;
    result.Delta_delta_fixed = Delta_delta_fixed;
    result.R_delta_min = R_delta_min;
    result.Delta_delta_min = Delta_delta_min;
    result.R_delta_pair = R_delta_pair;
    result.Delta_delta_pair = Delta_delta_pair;
    result.R_delta_adapt_pair = R_delta_adapt_pair;
    result.Delta_delta_adapt_pair = Delta_delta_adapt_pair;

    result.functional_true = functional_true;
    result.functional_delta = functional_delta;
    result.true_local_error = true_local_error;
    result.true_local_relative_error = true_local_relative_error;

    result.local_fixed = local_fixed;
    result.local_min = local_min;
    result.local_pair = local_pair;
    result.local_adapt_pair = local_adapt_pair;
    result.optimal_recovery = optimal_recovery;
    result.optimal_relative_error = optimal_relative_error;

    if ~isempty(local_fixed)
        result.posterior_fixed_relative = ...
            local_fixed.E_functional./max(abs(functional_true),eps);
    else
        result.posterior_fixed_relative = NaN(n_region,1);
    end

    if ~isempty(local_min)
        result.posterior_min_relative = ...
            local_min.E_functional./max(abs(functional_true),eps);
    else
        result.posterior_min_relative = NaN(n_region,1);
    end

    if ~isempty(local_pair)
        result.posterior_pair_relative = ...
            local_pair.E_functional./max(abs(functional_true),eps);
    else
        result.posterior_pair_relative = NaN(n_region,1);
    end

    if ~isempty(local_adapt_pair)
        result.posterior_adapt_pair_relative = ...
            local_adapt_pair.E_functional./max(abs(functional_true),eps);
    else
        result.posterior_adapt_pair_relative = NaN(n_region,1);
    end

    results(id) = result;

    % ---------------------------------------------------------------------
    % 6.7 输出诊断信息
    % ---------------------------------------------------------------------
    if par.verbose
        fprintf('细网格模型数据规模              = %d x %d\n',Nf,Nf);
        fprintf('反演网格规模                    = %d x %d\n',N,N);
        fprintf('细网格相对噪声                  = %.4f\n',fine_actual_noise);
        fprintf('粗网格相对噪声                  = %.4f\n',actual_noise);
        fprintf('Tikhonov 正则化参数 alpha       = %.6e\n',alpha_reg);
        fprintf('偏差原则目标值                  = %.4e\n',target_residual);
        fprintf('正则化解残差                    = %.4e\n',reg_residual);
        fprintf('Omega(z_delta)                  = %.4e\n',Omega_delta);
        fprintf('C_Omega                         = %.6f\n',C_Omega);
        fprintf('C_res                           = %.6f\n',C_residual);
        fprintf('C_min_raw                       = %.6f\n',C_min_raw);
        fprintf('固定 C                          = %.2f，真解可行 = %d\n',C_fixed,fixed_is_feasible);
        fprintf('最小可容许 C（向上保留两位）  = %.2f，真解可行 = %d\n',C_min,min_is_feasible);
        fprintf('固定双系数对 (C_res,C_Omega)    = (%.2f, %.2f)，真解可行 = %d\n', ...
            C_pair_residual,C_pair_omega,pair_is_feasible);
        fprintf('自适应双系数对 (C_res,C_Omega)  = (%.2f, %.2f)，真解可行 = %d\n', ...
            C_adapt_residual,C_adapt_omega,adapt_pair_is_feasible);
        fprintf('真实相对 L2 误差                = %.4e\n',result.true_relative_L2_error);

        for ir = 1:n_region
            fprintf('  %s: <ell,z_true>              = %.6e\n',regions(ir).name,functional_true(ir));
            fprintf('      <ell,z_delta>             = %.6e\n',functional_delta(ir));
            fprintf('      局部真实相对误差          = %.6e\n',true_local_relative_error(ir));

            if ~isempty(local_fixed)
                fprintf('      固定 C 后验相对误差       = %.6e\n',result.posterior_fixed_relative(ir));
            end
            if ~isempty(local_min)
                fprintf('      最小 C 后验相对误差       = %.6e\n',result.posterior_min_relative(ir));
            end
            if ~isempty(local_pair)
                fprintf('      固定双系数对后验相对误差 = %.6e\n',result.posterior_pair_relative(ir));
            end
            if ~isempty(local_adapt_pair)
                fprintf('      自适应双系数对后验相对误差 = %.6e\n',result.posterior_adapt_pair_relative(ir));
            end

            fprintf('      最优恢复相对误差          = %.6e\n',optimal_relative_error(ir));
            fprintf('      最优恢复 t=mu/lambda      = %.6e\n',optimal_recovery(ir).t_ratio);
        end

        if ~isempty(local_fixed)
            fprintf('固定 C：Algorithm 2 最大约束违背 = %.3e，最大相对对偶间隙 = %.3e\n', ...
                local_fixed.max_constraint_violation,local_fixed.max_relative_duality_gap);
        end
        if ~isempty(local_min)
            fprintf('最小 C：Algorithm 2 最大约束违背 = %.3e，最大相对对偶间隙 = %.3e\n', ...
                local_min.max_constraint_violation,local_min.max_relative_duality_gap);
        end
        if ~isempty(local_pair)
            fprintf('固定双系数对：Algorithm 2 最大约束违背 = %.3e，最大相对对偶间隙 = %.3e\n', ...
                local_pair.max_constraint_violation,local_pair.max_relative_duality_gap);
        end
        if ~isempty(local_adapt_pair)
            fprintf('自适应双系数对：Algorithm 2 最大约束违背 = %.3e，最大相对对偶间隙 = %.3e\n', ...
                local_adapt_pair.max_constraint_violation,local_adapt_pair.max_relative_duality_gap);
        end

        fprintf('============================================================\n');

        if ~fixed_is_feasible && par.warn_if_true_not_feasible
            warning(['固定 C 对应的后验放大因子过小，', ...
                '无法使离散精确解包含在后验可行集合中。']);
        end
    end
end

%% ========================================================================
%  7. 精确位场及 D1+D2 组合局部区域
% =========================================================================
figure('Color','w','Name','例8.6：精确位场与 D1+D2 局部区域');
imagesc(x_roi,x_roi,z_true_roi);
axis image xy;
colorbar;
hold on;

region_colors = [0.8500 0.3250 0.0980; 0.4940 0.1840 0.5560];

rectangle('Position',[b1(1),b1(3),b1(2)-b1(1),b1(4)-b1(3)], ...
    'EdgeColor',region_colors(1,:),'LineWidth',1.8,'LineStyle','--');
rectangle('Position',[b2(1),b2(3),b2(2)-b2(1),b2(4)-b2(3)], ...
    'EdgeColor',region_colors(2,:),'LineWidth',1.8,'LineStyle','--');

text(b1(1),b1(4),'  D_1','Color','w','FontWeight','bold','FontSize',11,'VerticalAlignment','bottom');
text(b2(1),b2(4),'  D_2','Color','w','FontWeight','bold','FontSize',11,'VerticalAlignment','bottom');

hold off;
xlabel('x_1');
ylabel('x_2');
title('精确位场及组合局部区域 D_1+D_2');

%% ========================================================================
%  8. 后验可行集合放大系数 C 的比较
% =========================================================================
delta_plot = [results.delta].';
C_omega_plot = [results.C_Omega].';
C_residual_plot = [results.C_residual].';
C_min_plot = [results.C_min].';
C_fixed_plot = [results.C_fixed].';

% 统一绘图配色
color_true  = [0.0000 0.4470 0.7410];      % 蓝色：真实误差
color_fixed = [0.8500 0.3250 0.0980];      % 橙色：固定单常数 C
color_pair  = [0.0000 0.6200 0.4510];      % 绿色：固定双系数对
color_adapt_pair = [0.4940 0.1840 0.5560]; % 紫色：自适应双系数对
color_min   = [0.8000 0.0000 0.0000];      % 红色：可容许单常数 C_min
color_opt   = [0.20 0.20 0.20];            % 深灰：最优恢复误差
color_bad   = [0.10 0.10 0.10];            % 黑色：不可行标记

figure('Color','w','Name','例8.6：后验可行集合放大系数 C');

plot(delta_plot,C_residual_plot, ...
    'Color',[0.30 0.30 0.30],'Marker','v','LineStyle','-', ...
    'LineWidth',1.5,'MarkerSize',6,'DisplayName','C_{res}');
hold on;

plot(delta_plot,C_omega_plot, ...
    'Color',color_fixed,'Marker','o','LineStyle','--', ...
    'LineWidth',1.5,'MarkerSize',6,'DisplayName','C_{\Omega}');

plot(delta_plot,C_min_plot, ...
    'Color',color_true,'Marker','s','LineStyle','-', ...
    'LineWidth',1.8,'MarkerSize',6,'DisplayName','C_{min}');

plot(delta_plot,C_fixed_plot, ...
    'Color',[0.55 0.55 0.55],'Marker','none','LineStyle',':', ...
    'LineWidth',1.5,'DisplayName',sprintf('固定 C=%.2f',par.C_posterior));

hold off; grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('可行集合放大系数');
legend('Location','northwest','Interpreter','tex');
xlim([delta_plot(1),delta_plot(end)]);
xticks(delta_plot);
set(gca,'FontSize',11);

%% ========================================================================
%  9. 局部真实误差、后验误差和最优恢复误差
% =========================================================================

figure('Color','w','Name','例8.6：局部误差、后验估计与最优恢复误差');

if n_region==1
    tiledlayout(1,1,'TileSpacing','compact','Padding','compact');
else
    tiledlayout(1,n_region,'TileSpacing','compact','Padding','compact');
end

for ir = 1:n_region
    true_rel = zeros(n_delta,1);
    post_fixed_rel = NaN(n_delta,1);
    post_min_rel = NaN(n_delta,1);
    post_pair_rel = NaN(n_delta,1);
    post_adapt_pair_rel = NaN(n_delta,1);
    opt_rel = zeros(n_delta,1);

    fixed_feasible_plot = false(n_delta,1);
    pair_feasible_plot = false(n_delta,1);
    adapt_pair_feasible_plot = false(n_delta,1);

    for id = 1:n_delta
        true_rel(id) = results(id).true_local_relative_error(ir);
        post_fixed_rel(id) = results(id).posterior_fixed_relative(ir);
        post_min_rel(id) = results(id).posterior_min_relative(ir);
        post_pair_rel(id) = results(id).posterior_pair_relative(ir);
        post_adapt_pair_rel(id) = results(id).posterior_adapt_pair_relative(ir);
        opt_rel(id) = results(id).optimal_relative_error(ir);

        fixed_feasible_plot(id) = results(id).fixed_is_feasible;
        pair_feasible_plot(id) = results(id).pair_is_feasible;
        adapt_pair_feasible_plot(id) = results(id).adapt_pair_is_feasible;
    end

    nexttile;

    plot(delta_plot,true_rel,'Color',color_true,'Marker','^','LineStyle','-', ...
        'LineWidth',1.5,'MarkerSize',7,'DisplayName','近似解局部相对误差');
    hold on;

    plot(delta_plot,opt_rel,'Color',color_opt,'Marker','d','LineStyle','--', ...
        'LineWidth',1.6,'MarkerSize',6,'DisplayName','最优恢复误差');

    if strcmpi(par.C_mode,'fixed') || strcmpi(par.C_mode,'both')
        plot(delta_plot,post_fixed_rel,'Color',color_fixed,'Marker','s','LineStyle',':', ...
            'LineWidth',1.5,'MarkerSize',7, ...
            'DisplayName',sprintf('固定 C_{res}=C_{\\Omega}=%.2f 的局部后验估计',par.C_posterior));
    end

    if strcmpi(par.C_mode,'min') || strcmpi(par.C_mode,'both')
        plot(delta_plot,post_min_rel,'Color',color_min,'Marker','s','LineStyle','-', ...
            'LineWidth',1.6,'MarkerSize',7,'DisplayName','可容许 C_{min} 的局部后验估计');
    end

    if par.use_pair_C
        plot(delta_plot,post_pair_rel,'Color',color_pair,'Marker','o','LineStyle',':', ...
            'LineWidth',1.5,'MarkerSize',7, ...
            'DisplayName',sprintf('固定 (C_{res},C_{\\Omega})=(%.2f,%.2f) 的局部后验估计', ...
            par.C_pair_residual,par.C_pair_omega));
    end

    if par.use_adapt_pair_C
        plot(delta_plot,post_adapt_pair_rel,'Color',color_adapt_pair,'Marker','o','LineStyle','-', ...
            'LineWidth',1.6,'MarkerSize',7,'DisplayName','可容许 (C_{res},C_{\Omega}) 的局部后验估计');
    end

    % 在对应曲线上用黑色 x 标出“真解不在后验可行集合中”的点。
    bad_fixed = ~fixed_feasible_plot;
    if any(bad_fixed)
        plot(delta_plot(bad_fixed),post_fixed_rel(bad_fixed),'Color',color_bad,'Marker','x','LineStyle','none', ...
            'LineWidth',1.8,'MarkerSize',8,'DisplayName','真解不在固定 C 可行集合中');
    end

    bad_pair = ~pair_feasible_plot;
    if any(bad_pair)
        plot(delta_plot(bad_pair),post_pair_rel(bad_pair),'Color',color_bad,'Marker','x','LineStyle','none', ...
            'LineWidth',1.8,'MarkerSize',8,'DisplayName','真解不在固定系数对可行集合中');
    end

    hold off; grid on; box on;
    xlabel('相对噪声水平 \delta');
    ylabel('相对误差');
    title('D_1+D_2 局部后验误差估计');
    legend('Location','northwest','Interpreter','tex');
    xlim([delta_plot(1),delta_plot(end)]);
    xticks(delta_plot);
    set(gca,'FontSize',11);
end

%% ========================================================================
%  9.1 delta_show 时对角线上的逐点后验上下界
% =========================================================================

[~,id_show] = min(abs(delta_plot-par.delta_show));
delta_show = delta_plot(id_show);

if abs(delta_show-par.delta_show)>1.0e-12
    warning('par.delta_show 不在 delta_list 中，改用最近的 delta=%.3f。',delta_show);
end

r_show = results(id_show);

% -------------------------------------------------------------------------
% 重建 delta_show 对应的含噪观测数据
% -------------------------------------------------------------------------
noise_scale_show = delta_show*grid_norm(u_true)/noise_direction_coarse_norm;

noise_fine_show = noise_scale_show*noise_direction_fine;

u_delta_fine_show = u_true_fine+noise_fine_show;

u_delta_show = u_delta_fine_show(1:stride:end,1:stride:end);

udelta_hat_show = fft2(u_delta_show);

% delta_show 对应的 Tikhonov 近似解
z_delta_show = r_show.z_delta;
z_delta_hat_show = fft2(z_delta_show);

% -------------------------------------------------------------------------
% 对角线 x1=x2 上的所有粗网格点
% -------------------------------------------------------------------------
diag_id = (1:N).';
diag_indices = sub2ind([N,N],diag_id,diag_id);

x_diag = x(:);

% 精确解与 Tikhonov 近似解的对角线截面
z_true_diag = diag(z_true);
z_delta_diag = diag(z_delta_show);

fprintf('\n');
fprintf('============================================================\n');
fprintf('delta=%.2f：计算对角线逐点后验上下界\n',delta_show);
fprintf('============================================================\n');

% 1. 固定单常数 C
fprintf('固定 C=%.2f：计算逐点上下界...\n',r_show.C_fixed);

point_fixed = algorithm2_pointwise_fourier_fast(Ahat,Rhat,udelta_hat_show,z_delta_hat_show, ...
    r_show.R_delta_fixed,r_show.Delta_delta_fixed,diag_indices,N,par);


% 2. 固定双系数对 (C_res,C_Omega)
fprintf('固定双系数对 (C_res,C_Omega)=(%.2f,%.2f)：计算逐点上下界...\n', ...
    r_show.C_pair_residual,r_show.C_pair_omega);

point_pair = algorithm2_pointwise_fourier_fast(Ahat,Rhat,udelta_hat_show,z_delta_hat_show, ...
    r_show.R_delta_pair,r_show.Delta_delta_pair,diag_indices,N,par);


% 3. 自适应双系数对 (C_res,C_Omega)
fprintf('可容许双系数对 (C_res,C_Omega)=(%.2f,%.2f)：计算逐点上下界...\n', ...
    r_show.C_adapt_residual,r_show.C_adapt_omega);

point_adapt_pair = algorithm2_pointwise_fourier_fast(Ahat,Rhat,udelta_hat_show,z_delta_hat_show, ...
    r_show.R_delta_adapt_pair,r_show.Delta_delta_adapt_pair,diag_indices,N,par);

% 4. 可容许单常数 C_min
fprintf('可容许 C_min=%.2f：计算逐点上下界...\n',r_show.C_min);

point_min = algorithm2_pointwise_fourier_fast(Ahat,Rhat,udelta_hat_show,z_delta_hat_show, ...
    r_show.R_delta_min,r_show.Delta_delta_min,diag_indices,N,par);

% 以 Tikhonov 近似解为基准构造非对称双边后验区间
upper_fixed = z_delta_diag(:)+point_fixed.E_plus(:);
lower_fixed = z_delta_diag(:)-point_fixed.E_minus(:);


upper_pair = z_delta_diag(:)+point_pair.E_plus(:);
lower_pair = z_delta_diag(:)-point_pair.E_minus(:);

upper_adapt_pair = z_delta_diag(:)+point_adapt_pair.E_plus(:);
lower_adapt_pair = z_delta_diag(:)-point_adapt_pair.E_minus(:);


upper_min = z_delta_diag(:)+point_min.E_plus(:);
lower_min =z_delta_diag(:)-point_min.E_minus(:);

% 数值一致性检查：upper = zmax, lower = zmin
tol_bound = 1.0e-8;

err_fixed_upper = max(abs(upper_fixed-point_fixed.zmax(:)));

err_fixed_lower = max(abs(lower_fixed-point_fixed.zmin(:)));

err_pair_upper = max(abs(upper_pair-point_pair.zmax(:)));

err_pair_lower = max(abs(lower_pair-point_pair.zmin(:)));

err_adapt_upper = max(abs(upper_adapt_pair-point_adapt_pair.zmax(:)));

err_adapt_lower = max(abs(lower_adapt_pair-point_adapt_pair.zmin(:)));

err_min_upper = max(abs(upper_min-point_min.zmax(:)));

err_min_lower = max(abs(lower_min-point_min.zmin(:)));

if max([err_fixed_upper,err_fixed_lower,err_pair_upper,err_pair_lower, ...
        err_adapt_upper,err_adapt_lower,err_min_upper,err_min_lower]) > tol_bound

    warning(['由 E_+/E_- 构造的上下界与直接计算的 zmax/zmin ', ...
        '存在超过容差的差异，请检查后验集合可行性。']);
end

fprintf('对角线逐点后验上下界计算完成。\n');
fprintf('============================================================\n');


%% ========================================================================
%  9.2 固定单常数 C：对角线逐点后验上下界
% =========================================================================

figure('Color','w','Name',sprintf('delta=%.2f 固定 C 对角线逐点后验区间',delta_show));

% 后验上下界之间的区域
fill([x_diag;flipud(x_diag)],[lower_fixed;flipud(upper_fixed)],color_fixed, ...
    'FaceAlpha',0.18,'EdgeColor','none','DisplayName','逐点后验区间');

hold on;

% 上界：z_delta + E_+
plot(x_diag,upper_fixed,'Color',color_fixed,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','上界 z_\delta+E_+');

% 下界：z_delta - E_-
plot(x_diag,lower_fixed,'Color',color_fixed,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','下界 z_\delta-E_-');

% 精确解
plot(x_diag,z_true_diag,'Color',[0.9290 0.6940 0.1250],'LineStyle','-', ...
    'LineWidth',2.0,'DisplayName','精确解');

% Tikhonov 近似解
plot(x_diag,z_delta_diag,'Color',color_true,'LineStyle','-.','LineWidth',1.8, ...
    'DisplayName','Tikhonov 近似解');

hold off; grid on; box on;

xlabel('x_1=x_2');
ylabel('z(x_1,x_2)');

title(sprintf('固定 C=%.2f 的对角线逐点后验区间',r_show.C_fixed));

legend('Location','best','Interpreter','tex');

xlim([x_diag(1),x_diag(end)]);

set(gca,'FontSize',11);

%% ========================================================================
%  9.3 固定双系数对：对角线逐点后验上下界
% =========================================================================

figure('Color','w','Name',sprintf('delta=%.2f 固定双系数对对角线逐点后验区间',delta_show));

fill([x_diag;flipud(x_diag)],[lower_pair;flipud(upper_pair)],color_pair,'FaceAlpha',0.18, ...
    'EdgeColor','none','DisplayName','逐点后验区间');

hold on;

% 上界：z_delta + E_+
plot(x_diag,upper_pair,'Color',color_pair,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','上界 z_\delta+E_+');

% 下界：z_delta - E_-
plot(x_diag,lower_pair,'Color',color_pair,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','下界 z_\delta-E_-');

% 精确解
plot(x_diag,z_true_diag,'Color',[0.9290 0.6940 0.1250],'LineStyle','-', ...
    'LineWidth',2.0,'DisplayName','精确解');

% Tikhonov 近似解
plot(x_diag,z_delta_diag,'Color',color_true,'LineStyle','-.','LineWidth',1.8, ...
    'DisplayName','Tikhonov 近似解');

hold off; grid on; box on;

xlabel('x_1=x_2');
ylabel('z(x_1,x_2)');

title(sprintf(['固定系数对 (C_{res},C_{\\Omega})=','(%.2f,%.2f) 的对角线逐点后验区间'], ...
    r_show.C_pair_residual, r_show.C_pair_omega),'Interpreter','tex');

legend('Location','best','Interpreter','tex');

xlim([x_diag(1),x_diag(end)]);

set(gca,'FontSize',11);

%% ========================================================================
%  9.4 自适应双系数对：对角线逐点后验上下界
% =========================================================================

figure('Color','w','Name',sprintf('delta=%.2f 自适应双系数对对角线逐点后验区间',delta_show));

fill([x_diag;flipud(x_diag)],[lower_adapt_pair;flipud(upper_adapt_pair)],color_adapt_pair, ...
    'FaceAlpha',0.18,'EdgeColor','none','DisplayName','逐点后验区间');

hold on;

% 上界：z_delta + E_+
plot(x_diag,upper_adapt_pair,'Color',color_adapt_pair,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','上界 z_\delta+E_+');

% 下界：z_delta - E_-
plot(x_diag,lower_adapt_pair,'Color',color_adapt_pair,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','下界 z_\delta-E_-');

% 精确解
plot(x_diag,z_true_diag,'Color',[0.9290 0.6940 0.1250],'LineStyle','-', ...
    'LineWidth',2.0,'DisplayName','精确解');

% Tikhonov 近似解
plot(x_diag,z_delta_diag,'Color',color_true,'LineStyle','-.','LineWidth',1.8, ...
    'DisplayName','Tikhonov 近似解');

hold off; grid on; box on;

xlabel('x_1=x_2');
ylabel('z(x_1,x_2)');

title(sprintf(['可容许系数对 (C_{res},C_{\\Omega})=','(%.2f,%.2f) 的对角线逐点后验区间'], ...
    r_show.C_adapt_residual,r_show.C_adapt_omega),'Interpreter','tex');

legend('Location','best','Interpreter','tex');

xlim([x_diag(1),x_diag(end)]);

set(gca,'FontSize',11);

%% ========================================================================
%  9.5 可容许单常数 C_min：对角线逐点后验上下界
% =========================================================================

figure('Color','w','Name',sprintf('delta=%.2f Cmin 对角线逐点后验区间',delta_show));

fill([x_diag;flipud(x_diag)],[lower_min;flipud(upper_min)],color_min, ...
    'FaceAlpha',0.18,'EdgeColor','none','DisplayName','逐点后验区间');

hold on;

% 上界：z_delta + E_+
plot(x_diag,upper_min,'Color',color_min,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','上界 z_\delta+E_+');

% 下界：z_delta - E_-
plot(x_diag,lower_min,'Color',color_min,'LineStyle','--','LineWidth',1.5, ...
    'DisplayName','下界 z_\delta-E_-');

% 精确解
plot(x_diag,z_true_diag,'Color',[0.9290 0.6940 0.1250],'LineStyle','-', ...
    'LineWidth',2.0,'DisplayName','精确解');

% Tikhonov 近似解
plot(x_diag,z_delta_diag,'Color',color_true,'LineStyle','-.','LineWidth',1.8, ...
    'DisplayName','Tikhonov 近似解');

hold off; grid on; box on;

xlabel('x_1=x_2');
ylabel('z(x_1,x_2)');

title(sprintf('可容许 C_{min}=%.2f 的对角线逐点后验区间',r_show.C_min),'Interpreter','tex');

legend('Location','best','Interpreter','tex');

xlim([x_diag(1),x_diag(end)]);

set(gca,'FontSize',11);

%% ========================================================================
%  10. 可选：绘制源函数
% =========================================================================
if par.plot_source
    figure('Color','w','Name','例8.6：源函数');
    imagesc(xf,xf,w_fine);
    axis image xy;
    colorbar;
    xlabel('\xi_1');
    ylabel('\xi_2');
    title('w=20\chi_{T_1}+\chi_{T_2}');
end

%% ========================================================================
%  11. 输出 D1+D2 组合局部泛函的数值结果表
% =========================================================================
for id = 1:n_delta
    r = results(id);

    fprintf('\n');
    fprintf('delta = %.4f：C_fixed=%.2f，C_min=%.2f，pair=(%.2f,%.2f)，adapt=(%.2f,%.2f)\n', ...
        r.delta,r.C_fixed,r.C_min,r.C_pair_residual,r.C_pair_omega, ...
        r.C_adapt_residual,r.C_adapt_omega);
    fprintf([' 区域     真值泛函       近似泛函       真实相对误差', ...
        '    固定C后验   Cmin后验   固定双系数后验   自适应双系数后验   最优恢复\n']);
    fprintf(['----------------------------------------------------------------', ...
        '--------------------------------------------------------------------------\n']);

    for ir = 1:n_region
        fprintf('%4s  %12.5e  %12.5e  %12.5e  %12.5e  %12.5e  %16.5e  %18.5e  %12.5e\n', ...
            regions(ir).name, r.functional_true(ir), r.functional_delta(ir), r.true_local_relative_error(ir), ...
            r.posterior_fixed_relative(ir), r.posterior_min_relative(ir), r.posterior_pair_relative(ir), ...
            r.posterior_adapt_pair_relative(ir), r.optimal_relative_error(ir));
    end
end

%% ========================================================================
%  根据偏差原则选取正则化参数
% =========================================================================
function alpha = choose_tikhonov_alpha(Ahat,Rhat,uhat,N,target)
    residual = @(a) tikhonov_residual(a,Ahat,Rhat,uhat,N);

    alo = 1.0e-100;
    ahi = 1.0e2;

    while residual(alo)>target && alo>realmin
        alo = alo/10;
    end
    while residual(ahi)<target && ahi<1.0e100
        ahi = ahi*10;
    end

    if residual(alo)>target
        error('未找到偏差方程根的左端括区间。');
    end
    if residual(ahi)<target
        error('未找到偏差方程根的右端括区间。');
    end

    for k = 1:50
        amid = sqrt(alo*ahi);
        if residual(amid)<target
            alo = amid;
        else
            ahi = amid;
        end
    end

    alpha = sqrt(alo*ahi);
end

function value = tikhonov_residual(alpha,Ahat,Rhat,uhat,N)
    zhat = conj(Ahat).*uhat./(abs(Ahat).^2+alpha*Rhat);
    value = spectral_norm(Ahat.*zhat-uhat,N);
end

%% ========================================================================
%  算法2的区域平均泛函 Fourier 实现
% =========================================================================
function local = algorithm2_region_fourier_fast(Ahat,Rhat,uhat,zeta_hat,R2,Delta,regions,N,par)

    n_region = numel(regions);
    n = N^2;

    bhat = conj(Ahat).*uhat;
    u2 = real(sum(abs(uhat(:)).^2))/n;

    logt_grid = linspace(par.dual_logt_min,par.dual_logt_max,par.dual_grid_size);
    ng = numel(logt_grid);

    values_plus = inf(n_region,ng);
    values_minus = inf(n_region,ng);

    chat = cell(n_region,1);
    for ir = 1:n_region
        chat{ir} = fft2(regions(ir).weights);
    end

    for kg = 1:ng
        [center,Mhat,a,valid] = support_region_components( ...
            logt_grid(kg),Ahat,Rhat,bhat,u2,R2,Delta,N);

        if ~valid
            continue;
        end

        for ir = 1:n_region
            cMc = real(sum(abs(chat{ir}(:)).^2./Mhat(:)))/n;
            if ~isfinite(cMc) || cMc<=0
                continue;
            end

            q = sqrt(a*cMc);
            center_value = sum(regions(ir).weights(:).*center(:));

            values_plus(ir,kg) = center_value+q;
            values_minus(ir,kg) = -center_value+q;
        end
    end

    fmax = zeros(n_region,1);
    fmin = zeros(n_region,1);
    E_plus = zeros(n_region,1);
    E_minus = zeros(n_region,1);
    E_functional = zeros(n_region,1);

    % 保存两个单侧极值对应的候选解，以及最终后验误差对应的最坏候选解
    zmax_candidate = cell(n_region,1);
    zmin_candidate = cell(n_region,1);
    z_worst_candidate = cell(n_region,1);
    worst_side = zeros(n_region,1);

    diagnostics = repmat(empty_local_diagnostic(),2*n_region,1);
    max_constraint_violation = 0;
    max_relative_duality_gap = 0;

    zeta = real(ifft2(zeta_hat));

    for ir = 1:n_region
        [upper,logt_upper,diag_upper] = refine_support_region_value( ...
            +1,regions(ir),values_plus(ir,:),logt_grid, ...
            Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,par);

        [negative_lower,logt_lower,diag_lower] = refine_support_region_value( ...
            -1,regions(ir),values_minus(ir,:),logt_grid, ...
            Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,par);

        % 根据最优对偶参数恢复两个单侧极值对应的空间域候选解
        candidate_upper = recover_support_region_candidate( ...
            logt_upper,+1,regions(ir),Ahat,Rhat,bhat,uhat,u2,R2,Delta,N);
        candidate_lower = recover_support_region_candidate( ...
            logt_lower,-1,regions(ir),Ahat,Rhat,bhat,uhat,u2,R2,Delta,N);

        zmax_candidate{ir} = candidate_upper.z_candidate;
        zmin_candidate{ir} = candidate_lower.z_candidate;

        fmax(ir) = upper;
        fmin(ir) = -negative_lower;

        fdelta = sum(regions(ir).weights(:).*zeta(:));
        E_plus(ir) = max(fmax(ir)-fdelta,0);
        E_minus(ir) = max(fdelta-fmin(ir),0);
        E_functional(ir) = max(E_plus(ir),E_minus(ir));

        % 选择使 |ell(z)-ell(z_delta)| 最大的一侧作为“后验误差对应解”
        if E_plus(ir)>=E_minus(ir)
            z_worst_candidate{ir} = zmax_candidate{ir};
            worst_side(ir) = +1;
        else
            z_worst_candidate{ir} = zmin_candidate{ir};
            worst_side(ir) = -1;
        end

        diag_upper.logt = logt_upper;
        diag_upper.t = exp(logt_upper);
        diag_lower.logt = logt_lower;
        diag_lower.t = exp(logt_lower);
        diag_upper.index = ir;
        diag_lower.index = ir;

        diagnostics(2*ir-1) = diag_upper;
        diagnostics(2*ir) = diag_lower;

        max_constraint_violation = max(max_constraint_violation, ...
            max(diag_upper.constraint_violation,diag_lower.constraint_violation));
        max_relative_duality_gap = max(max_relative_duality_gap, ...
            max(diag_upper.relative_duality_gap,diag_lower.relative_duality_gap));
    end

    local = struct();
    local.fmax = fmax;
    local.fmin = fmin;
    local.E_plus = E_plus;
    local.E_minus = E_minus;
    local.E_functional = E_functional;
    local.zmax_candidate = zmax_candidate;
    local.zmin_candidate = zmin_candidate;
    local.z_worst_candidate = z_worst_candidate;
    local.worst_side = worst_side;
    local.max_constraint_violation = max_constraint_violation;
    local.max_relative_duality_gap = max_relative_duality_gap;
    local.diagnostics = diagnostics;
end

%% ========================================================================
%  区域平均泛函：单侧支撑函数的一维对偶精化
% =========================================================================
function [support_value,best_logt,diagInfo] = refine_support_region_value( ...
    sign_value,region,coarse_values,logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,par)

    [coarse_best,idx] = min(coarse_values);
    if ~isfinite(coarse_best)
        error('区域局部后验误差的对偶扫描未得到有限值。');
    end

    il = max(1,idx-1);
    ir = min(numel(logt_grid),idx+1);
    left = logt_grid(il);
    right = logt_grid(ir);

    objective = @(p) support_region_objective( ...
        p,sign_value,region,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if left==right
        best_logt = logt_grid(idx);
        support_value = coarse_best;
    else
        [best_logt,support_value] = fminbnd(objective,left,right, ...
            optimset('Display','off','TolX',par.dual_tol_x,'MaxIter',250));

        if coarse_best<support_value
            best_logt = logt_grid(idx);
            support_value = coarse_best;
        end
    end

    candidate = recover_support_region_candidate( ...
        best_logt,sign_value,region,Ahat,Rhat,bhat,uhat,u2,R2,Delta,N);

    fdelta = sum(region.weights(:).*zeta(:));
    feasible_lower = sign_value*fdelta;

    if candidate.constraint_violation<=par.dual_feas_tol
        primal_lower = max(feasible_lower,candidate.primal_value);
    else
        primal_lower = feasible_lower;
    end

    relative_gap = max(support_value-primal_lower,0)/max([1,abs(support_value),abs(primal_lower)]);

    diagInfo = empty_local_diagnostic();
    diagInfo.sign = sign_value;
    diagInfo.dual_value = support_value;
    diagInfo.primal_value = candidate.primal_value;
    diagInfo.primal_lower_bound = primal_lower;
    diagInfo.relative_duality_gap = relative_gap;
    diagInfo.omega_ratio = candidate.omega/R2;
    diagInfo.residual_ratio = candidate.residual/Delta;
    diagInfo.constraint_violation = candidate.constraint_violation;
end

%% ========================================================================
%  区域平均泛函：固定 t=mu/lambda 时的公共量
% =========================================================================
function [center,Mhat,a,valid] = support_region_components(logt,Ahat,Rhat,bhat,u2,R2,Delta,N)

    center = [];
    Mhat = [];
    a = NaN;
    valid = false;

    if ~isfinite(logt)
        return;
    end

    t = exp(logt);
    Mhat = abs(Ahat).^2+t*Rhat;

    if any(~isfinite(Mhat(:))) || any(Mhat(:)<=0)
        return;
    end

    n = N^2;
    bMb = real(sum(abs(bhat(:)).^2./Mhat(:)))/n;
    a = bMb-u2+n*Delta^2+t*n*R2;

    if ~isfinite(a) || a<=0
        return;
    end

    center = real(ifft2(bhat./Mhat));
    valid = true;
end

%% ========================================================================
%  区域平均泛函：单个对偶目标函数
% =========================================================================
function value = support_region_objective(logt,sign_value,region,Ahat,Rhat,bhat,u2,R2,Delta,N)

    [center,Mhat,a,valid] = support_region_components(logt,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if ~valid
        value = Inf;
        return;
    end

    n = N^2;
    chat = fft2(region.weights);
    cMc = real(sum(abs(chat(:)).^2./Mhat(:)))/n;

    if ~isfinite(cMc) || cMc<=0
        value = Inf;
        return;
    end

    q = sqrt(a*cMc);
    center_value = sum(region.weights(:).*center(:));
    value = sign_value*center_value+q;
end

%% ========================================================================
%  区域平均泛函：由对偶比例恢复原问题候选解
% =========================================================================
function candidate = recover_support_region_candidate(logt,sign_value,region,Ahat,Rhat,bhat,uhat,u2,R2,Delta,N)

    [~,Mhat,a,valid] = support_region_components(logt,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if ~valid
        error('恢复区域后验极值候选解时出现无效对偶参数。');
    end

    n = N^2;
    chat = sign_value*fft2(region.weights);
    cMc = real(sum(abs(chat(:)).^2./Mhat(:)))/n;

    if cMc<=0
        error('恢复区域后验极值候选解时出现非正二次型。');
    end

    scale = sqrt(a/cMc);
    zhat = (bhat+scale*chat)./Mhat;

    z_candidate = real(ifft2(zhat));
    primal_value = sign_value*sum(region.weights(:).*z_candidate(:));

    omega = spectral_omega(zhat,Rhat,N);
    residual = spectral_norm(Ahat.*zhat-uhat,N);

    violation_omega = max(omega/R2-1,0);
    violation_residual = max(residual/Delta-1,0);

    candidate = struct();
    candidate.primal_value = primal_value;
    candidate.omega = omega;
    candidate.residual = residual;
    candidate.constraint_violation = max(violation_omega,violation_residual);
    candidate.z_candidate = z_candidate;
end

%% ========================================================================
%  Bayev Lagrange 原理：有限维最优恢复误差
% =========================================================================
function out = optimal_recovery_error_fourier(Ahat,Rhat,weights,R_prior,Delta_info,N,par)
    % 相关极值问题：
    %   max <ell,z>
    %   s.t. ||Az|| <= Delta_info,  Omega(z) <= R_prior.
    %
    % 对两个二次约束引入 Lagrange 乘子 lambda,mu，并令 t=mu/lambda，
    % 可将二维乘子搜索化为一维问题
    %   E_opt = min_{t>=0} sqrt((Delta_info^2+t*R_prior)*S(t)),
    % 其中 S(t)=sum |F[c]|^2/(|Ahat|^2+t*Rhat)，
    % c 为区域平均泛函的离散权重。

    chat = fft2(weights);
    logt_grid = linspace(par.opt_logt_min,par.opt_logt_max,par.opt_grid_size);

    values = zeros(size(logt_grid));
    for k = 1:numel(logt_grid)
        values(k) = optimal_recovery_objective(logt_grid(k),Ahat,Rhat,chat,R_prior,Delta_info);
    end

    [coarse_best,idx] = min(values);

    il = max(1,idx-1);
    ir = min(numel(logt_grid),idx+1);
    left = logt_grid(il);
    right = logt_grid(ir);

    objective = @(p) optimal_recovery_objective(p,Ahat,Rhat,chat,R_prior,Delta_info);

    if left==right
        best_logt = logt_grid(idx);
        best_value = coarse_best;
    else
        [best_logt,best_value] = fminbnd(objective,left,right,optimset('Display','off','TolX',par.opt_tol_x,'MaxIter',300));

        if coarse_best<best_value
            best_logt = logt_grid(idx);
            best_value = coarse_best;
        end
    end

    % 两个边界情形也参与比较：t=0（只由数据误差约束主导）和 t->Inf（只由先验约束主导）。
    den0 = abs(Ahat).^2;
    value_t0 = Delta_info*sqrt(real(sum(abs(chat(:)).^2./den0(:))));

    value_tinf = sqrt(R_prior*real(sum(abs(chat(:)).^2./Rhat(:))));

    [best_value,which_case] = min([best_value,value_t0,value_tinf]);

    if which_case==1
        t_ratio = exp(best_logt);
    elseif which_case==2
        t_ratio = 0;
        best_logt = -Inf;
    else
        t_ratio = Inf;
        best_logt = Inf;
    end

    out = empty_optimal_recovery();
    out.error = best_value;
    out.t_ratio = t_ratio;
    out.logt = best_logt;
    out.R_prior = R_prior;
    out.Delta_info = Delta_info;
    out.case_id = which_case;
end

function value = optimal_recovery_objective(logt,Ahat,Rhat,chat,R_prior,Delta_info)

    if ~isfinite(logt)
        value = Inf;
        return;
    end

    t = exp(logt);
    den = abs(Ahat).^2+t*Rhat;

    if any(~isfinite(den(:))) || any(den(:)<=0)
        value = Inf;
        return;
    end

    S = real(sum(abs(chat(:)).^2./den(:)));
    value = sqrt((Delta_info^2+t*R_prior)*S);
end

function out = empty_optimal_recovery()
    out = struct('error',NaN,'t_ratio',NaN,'logt',NaN,'R_prior',NaN, ...
        'Delta_info',NaN,'case_id',0);
end

%% ========================================================================
%  算法2的快速 Fourier 实现：分别计算两个单侧极值
% =========================================================================
function local = algorithm2_pointwise_fourier_fast(Ahat,Rhat,uhat,zeta_hat,R2,Delta,indices,N,par)

    n_point = numel(indices);    % 待估计点数
    n = N^2;                     % 二维未知量总数

    bhat = conj(Ahat).*uhat;
    u2 = real(sum(abs(uhat(:)).^2))/n;    % 经过当前 Fourier 归一化后的数据能量

    logt_grid = linspace(par.dual_logt_min,par.dual_logt_max,par.dual_grid_size);
    ng = numel(logt_grid);       % 粗搜索网格点数

    values_plus = inf(n_point,ng);
    values_minus = inf(n_point,ng);

    for kg = 1:ng
        % 后验集合在空间域的“中心”、点值极值中的半径项 q、当前参数是否有效
        [center,q,valid] = support_components(logt_grid(kg),Ahat,Rhat,bhat,u2,R2,Delta,N);
        if ~valid
            continue;
        end

        center_line = center(indices);        % 提取中心函数在所有选定点上的值
        values_plus(:,kg) = center_line+q;
        values_minus(:,kg) = -center_line+q;
    end

    zmax = zeros(n_point,1);
    zmin = zeros(n_point,1);
    E_plus = zeros(n_point,1);
    E_minus = zeros(n_point,1);
    E_pointwise = zeros(n_point,1);
    % 每个空间点有两个优化方向
    diagnostics = repmat(empty_local_diagnostic(),2*n_point,1);

    max_constraint_violation = 0;
    max_relative_duality_gap = 0;

    % 把传入的 Tikhonov Fourier 解恢复为空间域
    zeta = real(ifft2(zeta_hat));

    for ip = 1:n_point
        if par.verbose && ip == n_point
            fprintf('  算法2：正在计算第 %2d/%2d 个点\n',ip,n_point);
        end

        [upper,logt_upper,diag_upper] = refine_support_value( ...
            +1,indices(ip),values_plus(ip,:),logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,par);

        [negative_lower,logt_lower,diag_lower] = refine_support_value( ...
            -1,indices(ip),values_minus(ip,:),logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,par);

        zmax(ip) = upper;                % 恢复真正的上下界
        zmin(ip) = -negative_lower;

        zeta_i = zeta(indices(ip));      % 当前点的 Tikhonov 值
        E_plus(ip) = max(zmax(ip)-zeta_i,0);
        E_minus(ip) = max(zeta_i-zmin(ip),0);
        E_pointwise(ip) = max(E_plus(ip),E_minus(ip));

        diag_upper.logt = logt_upper;
        diag_upper.t = exp(logt_upper);
        diag_lower.logt = logt_lower;
        diag_lower.t = exp(logt_lower);

        diagnostics(2*ip-1) = diag_upper;
        diagnostics(2*ip) = diag_lower;

        max_constraint_violation = max(max_constraint_violation, ...
            max(diag_upper.constraint_violation, diag_lower.constraint_violation));
        max_relative_duality_gap = max(max_relative_duality_gap, ...
            max(diag_upper.relative_duality_gap,diag_lower.relative_duality_gap));
    end

    local = struct();
    local.zmax = zmax;
    local.zmin = zmin;
    local.E_plus = E_plus;
    local.E_minus = E_minus;
    local.E_pointwise = E_pointwise;
    local.max_constraint_violation = max_constraint_violation;
    local.max_relative_duality_gap = max_relative_duality_gap;
    local.diagnostics = diagnostics;
end

%% ========================================================================
%  单个空间点、单个符号方向：局部精化对偶支撑函数值
% =========================================================================
function [support_value,best_logt,diagInfo] = refine_support_value( ...
    sign_value,index,coarse_values,logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,par)

    [coarse_best,idx] = min(coarse_values);
    if ~isfinite(coarse_best)
        error('算法2的对偶扫描未得到有限值。');
    end

    % 选择最优粗网格点左右相邻位置，构造局部精化区间
    il = max(1,idx-1);
    ir = min(numel(logt_grid),idx+1);
    left = logt_grid(il);
    right = logt_grid(ir);

    % 一维目标函数
    objective = @(p) support_point_objective(p,sign_value,index,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if left==right
        best_logt = logt_grid(idx);
        support_value = coarse_best;
    else
        [best_logt,support_value] = fminbnd(objective,left,right, ...
            optimset('Display','off','TolX',par.dual_tol_x,'MaxIter',250));

        if coarse_best<support_value
            best_logt = logt_grid(idx);
            support_value = coarse_best;
        end
    end

    candidate = recover_support_candidate( ...
        best_logt,sign_value,index,Ahat,Rhat,bhat,uhat,u2,R2,Delta,N);

    feasible_lower = sign_value*zeta(index);
    % 若恢复出来的候选解满足约束，就把其目标值也用于构造原问题下界
    if candidate.constraint_violation<=par.dual_feas_tol
        primal_lower = max(feasible_lower,candidate.primal_value);
    else
        primal_lower = feasible_lower;
    end

    % 计算归一化对偶间隙
    relative_gap = max(support_value-primal_lower,0)/max([1,abs(support_value),abs(primal_lower)]);

    diagInfo = empty_local_diagnostic();
    diagInfo.sign = sign_value;
    diagInfo.index = index;
    diagInfo.dual_value = support_value;
    diagInfo.primal_value = candidate.primal_value;
    diagInfo.primal_lower_bound = primal_lower;
    diagInfo.relative_duality_gap = relative_gap;
    diagInfo.omega_ratio = candidate.omega/R2;
    diagInfo.residual_ratio = candidate.residual/Delta;
    diagInfo.constraint_violation = candidate.constraint_violation;
end

%% ========================================================================
%  固定 t=mu/lambda 时支撑函数的公共计算量
% =========================================================================
function [center,q,valid] = support_components(logt,Ahat,Rhat,bhat,u2,R2,Delta,N)

    center = [];
    q = Inf;
    valid = false;

    if ~isfinite(logt)
        return;
    end

    t = exp(logt);                 % 自动保证 t>0
    Mhat = abs(Ahat).^2+t*Rhat;    % 构造 Fourier 域对偶矩阵的对角元素
    if any(~isfinite(Mhat(:))) || any(Mhat(:)<=0)
        return;
    end

    n = N^2;
    bMb = real(sum(abs(bhat(:)).^2./Mhat(:)))/n;
    cMc = real(sum(1./Mhat(:)))/n;
    a = bMb-u2+n*Delta^2+t*n*R2;   % 对偶公式中的系数

    if ~isfinite(a) || ~isfinite(cMc) || a<=0 || cMc<=0
        return;
    end

    center = real(ifft2(bhat./Mhat));  % 计算固定 t 时后验可行集合的中心函数
    q = sqrt(a*cMc);                   % 计算点值支撑函数的半径项
    valid = isfinite(q);
end

%% ========================================================================
%  单个选定点对应的对偶目标函数
% =========================================================================
function value = support_point_objective(logt,sign_value,index,Ahat,Rhat,bhat,u2,R2,Delta,N)

    [center,q,valid] = support_components(logt,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if ~valid
        value = Inf;
        return;
    end

    value = sign_value*center(index)+q;
end

%% ========================================================================
%  根据最优对偶比例恢复原问题候选解
% =========================================================================
function candidate = recover_support_candidate(logt,sign_value,index,Ahat,Rhat,bhat,uhat,u2,R2,Delta,N)

    t = exp(logt);
    Mhat = abs(Ahat).^2+t*Rhat;
    n = N^2;

    bMb = real(sum(abs(bhat(:)).^2./Mhat(:)))/n;
    cMc = real(sum(1./Mhat(:)))/n;
    a = bMb-u2+n*Delta^2+t*n*R2;

    if a<=0 || cMc<=0
        error('恢复原问题候选解时出现了非正系数。');
    end

    c = zeros(N,N);
    c(index) = sign_value;
    chat = fft2(c);         % 把点值方向转换到 Fourier 域

    scale = sqrt(a/cMc);    % 计算 KKT 公式中的缩放因子
    zhat = (bhat+scale*chat)./Mhat;  % 根据最优性条件恢复极值候选解

    z_candidate = real(ifft2(zhat)); % 恢复空间域候选解
    primal_value = sign_value*z_candidate(index);

    omega = spectral_omega(zhat,Rhat,N);
    residual = spectral_norm(Ahat.*zhat-uhat,N);

    violation_omega = max(omega/R2-1,0);
    violation_residual = max(residual/Delta-1,0);

    candidate = struct();
    candidate.primal_value = primal_value;
    candidate.omega = omega;
    candidate.residual = residual;
    candidate.constraint_violation = max(violation_omega,violation_residual);
end

%% ========================================================================
%  Fourier 频率网格
% =========================================================================
function [k2,kabs] = fourier_frequencies_2d(N,L)
    if mod(N,2)==0
        modes = [0:N/2-1,-N/2:-1];
    else
        modes = [0:(N-1)/2,-(N-1)/2:-1];
    end

    k1 = (2*pi/L)*modes;
    [KX,KY] = meshgrid(k1,k1);
    k2 = KX.^2+KY.^2;
    kabs = sqrt(k2);
end

%% ========================================================================
%  初始化空的输出结构体
% =========================================================================
function result = empty_result()
    result = struct( ...
        'delta',NaN, ...
        'actual_noise',NaN, ...
        'alpha',NaN, ...
        'target_residual',NaN, ...
        'reg_residual',NaN, ...
        'Omega_delta',NaN, ...
        'true_Omega',NaN, ...
        'true_residual',NaN, ...
        'true_relative_L2_error',NaN, ...
        'z_delta',[], ...
        'C_Omega',NaN, ...
        'C_residual',NaN, ...
        'C_min_raw',NaN, ...
        'C_min',NaN, ...
        'C_fixed',NaN, ...
        'C_pair_residual',NaN, ...
        'C_pair_omega',NaN, ...
        'C_adapt_residual',NaN, ...
        'C_adapt_omega',NaN, ...
        'fixed_is_feasible',false, ...
        'min_is_feasible',false, ...
        'pair_is_feasible',false, ...
        'adapt_pair_is_feasible',false, ...
        'R_delta_fixed',NaN, ...
        'Delta_delta_fixed',NaN, ...
        'R_delta_min',NaN, ...
        'Delta_delta_min',NaN, ...
        'R_delta_pair',NaN, ...
        'Delta_delta_pair',NaN, ...
        'R_delta_adapt_pair',NaN, ...
        'Delta_delta_adapt_pair',NaN, ...
        'functional_true',[], ...
        'functional_delta',[], ...
        'true_local_error',[], ...
        'true_local_relative_error',[], ...
        'posterior_fixed_relative',[], ...
        'posterior_min_relative',[], ...
        'posterior_pair_relative',[], ...
        'posterior_adapt_pair_relative',[], ...
        'local_fixed',[], ...
        'local_min',[], ...
        'local_pair',[], ...
        'local_adapt_pair',[], ...
        'optimal_recovery',[], ...
        'optimal_relative_error',[]);
end

function d = empty_local_diagnostic()
    d = struct( ...
        'sign',0, ...
        'index',0, ...
        'logt',NaN, ...
        't',NaN, ...
        'dual_value',NaN, ...
        'primal_value',NaN, ...
        'primal_lower_bound',NaN, ...
        'relative_duality_gap',Inf, ...
        'omega_ratio',Inf, ...
        'residual_ratio',Inf, ...
        'constraint_violation',Inf);
end

%% ========================================================================
%  向上保留两位小数
% =========================================================================
function value = ceil_to_two_decimals(x)
    scaled = 100*x;
    nearest_integer = round(scaled);

    % 若 scaled 与整数仅有浮点舍入级别的差异，则认为原数已经只有两位小数，
    % 防止例如 1.0200000000000002 被错误进位成 1.03。
    tol = 1.0e-10*max(1,abs(scaled));
    if abs(scaled-nearest_integer)<=tol
        value = nearest_integer/100;
    else
        value = ceil(scaled)/100;
    end
end

%% ========================================================================
%  离散范数
% =========================================================================
function value = grid_norm(v)
    value = sqrt(mean(abs(v(:)).^2));
end

function value = spectral_norm(vhat,N)
    value = sqrt(real(sum(abs(vhat(:)).^2)))/N^2;
end

function value = spectral_omega(zhat,Rhat,N)
    value = real(sum(Rhat(:).*abs(zhat(:)).^2))/N^4;
end
