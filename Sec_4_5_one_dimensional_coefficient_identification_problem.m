% 4.5：一维椭圆方程非线性反系数问题
%
% 反问题：已知含噪状态观测 u^delta(x) 和精确右端项 f(x)，恢复正系数 a(x)：
%       -[a(x)u'(x)]' = f(x),  0 < x < 1,  u(0)=u(1)=0.
% 精确模型：  a_true(x) = x^2(1-x)^2 + 0.01,
%             u_true(x) = x^4(1-x)^4.
% 二阶 Tikhonov 正则化：
%       min_a ||F_n(a)-u_delta||_{L2}^2 + alpha ||a||_{W_2^2}^2,
% 正则化参数 alpha 由广义偏差原则选取
% 全局后验误差估计：
%       E(delta) = max_a ||a-a_delta||_{L2}
%       约束条件：Omega(a) <= C_omega Omega(a_delta),
%            ||F_n(a)-u_delta||_{L2}<= C_res ||F_n(a_delta)-u_delta||_{L2}.
%
% 比较四种放大系数规则：
%   1) 固定单值参数：C_res = C_omega = C_fix;
%      仅由 delta=0.005 校准，向上保留两位小数，随后保持不变.
%   2) 可容许单值参数：C_res = C_omega = C_min(delta),
%      每个噪声水平分别确定，使真解属于相应后验可行集合.
%   3) 固定双值参数：C_res = C_res_fix，C_omega = C_omega_fix,
%      两者仅由 delta=0.005 校准，随后保持不变.
%   4) 可容许双值参数：C_res = C_res_ad(delta)，C_omega = C_omega_ad(delta),
%      每个噪声水平分别确定.
% 说明：可容许系数以及固定系数的校准使用了已知真解信息,
% 运行需要 MATLAB Optimization Toolbox 中的 fmincon.
clc;clear;
close all;
%% ------------------------------ 可调参数 ------------------------------
n = 80;
x = linspace(0, 1, n).';
h = x(2) - x(1);

deltaList = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030];  % 相对噪声水平
numberDelta = numel(deltaList);
randomSeed = 20260728;
rng(randomSeed, 'twister');

% false: 用离散正演模型构造 f, 使 F_n(a_true)=u_true.
% true: 使用连续解析右端项 f, 此时会引入离散模型不一致误差.
useAnalyticF = false;

% 广义偏差原则常数
tauGDP = 1.01;

% 正系数 a(x) 的约束
lowerA = 1e-6;
upperA = 0.20;

% 全局后验极值搜索参数
numberPosteriorStarts = 24;
numberSensitivityDirections = 16;
numberRandomDirections = 30;
numberRayScanPoints = 81;

% 当前噪声水平覆盖 0.005--0.030, 适当放宽 alpha 的搜索范围.
alphaGrid = logspace(-3, -16, 27);
numberAlphaRefinements = 12;

% 可容许单值参数使用 max{1,C_res^ref,C_omega^ref};
% 可容许双值参数分别使用 max{1.01,C^ref}.
CsingleFloor = 1.00;
CdoubleFloor = 1.01;
ceil2 = @(z) ceil(100*z - 1e-12) / 100;
%% ----------------------- 离散 L2 与 W_2^2 范数 -----------------------
% ||a||_{W_2^2}^2 = ||a||_{L2}^2 + ||a''||_{L2}^2
w = h * ones(n, 1);
w([1, end]) = h / 2;
W = spdiags(w, 0, n, n);
D2 = sparse(1:n-2, 1:n-2, (1/h^2)*ones(n-2,1), n-2, n) ...
  + sparse(1:n-2, 2:n-1, (-2/h^2)*ones(n-2,1), n-2, n) ...
  + sparse(1:n-2, 3:n,    (1/h^2)*ones(n-2,1), n-2, n);

% Omega(a) = ||a||_{W_2^2}^2 = ||a||_{L2}^2 + ||a''||_{L2}^2。
% 因此 a'*Q*a 近似积分 int_0^1 [a^2 + (a'')^2] dx, 不包含一阶导数项.
Q = W + h*(D2.'*D2);
Q = 0.5*(Q + Q.');
%% ------------------------------ 精确模型 ------------------------------
g = x .* (1-x);
aTrue = g.^2 + 0.01;
uTrue = g.^4;
gp = 1 - 2*x;
aPrime = 2*g .* gp;
uPrime = 4*g.^3 .* gp;
uSecond = 12*g.^2 .* gp.^2 - 8*g.^3;
fAnalytic = -(aPrime .* uPrime + aTrue .* uSecond);
fAnalytic([1,end]) = 0;
if useAnalyticF
   f = fAnalytic;
else
   KTrue = diffusion_matrix(aTrue, h);
   f = zeros(n, 1);
   f(2:n-1) = KTrue * uTrue(2:n-1);
end
normATrue = l2norm_discrete(aTrue, w);
normUTrue = l2norm_discrete(uTrue, w);
omegaTrue = aTrue.' * Q * aTrue;
%% ---------------------- 固定高斯噪声方向 ---------------------------
% 噪声水平采用相同的随机扰动方向, 仅改变噪声幅值, 以减小随机性对比较的影响.
noiseDirection = randn(n, 1);
noiseDirection([1,end]) = 0;

% 将噪声方向与精确状态离散正交化, 以避免 ||u_delta|| 出现系统性尺度偏差.
noiseDirection = noiseDirection - uTrue * ((w.*uTrue).' * noiseDirection) / ((w.*uTrue).' * uTrue);
noiseDirection = noiseDirection / l2norm_discrete(noiseDirection, w);
%% =====================================================================
% 第 1 阶段: 计算不同噪声水平下的 Tikhonov 解以及真解参考系数 C_res^ref、C_omega^ref.
% ======================================================================
alphaChosen = zeros(numberDelta,1);
actualNoise = zeros(numberDelta,1);
residualRelative = zeros(numberDelta,1);
residualTargetRatio = zeros(numberDelta,1);
alphaBracketFound = false(numberDelta,1);
trueRelativeError = zeros(numberDelta,1);
omegaRatioTrue = zeros(numberDelta,1);  % C_omega^ref
resRatioTrue = zeros(numberDelta,1);    % C_res^ref
recoveredA = zeros(n, numberDelta);
uDeltaAll = zeros(n, numberDelta);
residualCenterAll = zeros(numberDelta,1);
omegaCenterAll = zeros(numberDelta,1);
parameterHistory = cell(numberDelta,1);
initialA = 0.04 * ones(n,1);
fprintf('\n===============================================================\n');
fprintf('非线性反系数算例：全局后验误差估计\n');
fprintf('n = %d, delta = [0.005 0.010 0.015 0.020 0.025 0.030]\n', n);
fprintf('广义偏差原则常数 tau_GDP = %.2f，useAnalyticF = %d\n', tauGDP, useAnalyticF);
fprintf('===============================================================\n');
fprintf('\n第 1 阶段：Tikhonov 重构与参考系数计算\n');
totalTimer = tic;
for k = 1:numberDelta
   delta = deltaList(k);

   % 相对噪声定义为 ||u_delta-u_true||/||u_true|| = delta.
   noiseAmplitude = delta * normUTrue;
   uDelta = uTrue + noiseAmplitude * noiseDirection;
   uDeltaAll(:,k) = uDelta;
   actualNoise(k) = l2norm_discrete(uDelta-uTrue, w) / normUTrue;
   targetResidual = tauGDP * delta * normUTrue;
   [aDelta, alpha, alphaInfo] = choose_alpha_by_discrepancy( ...
       initialA, uDelta, f, h, w, Q, lowerA, upperA,targetResidual, alphaGrid, numberAlphaRefinements);
   alphaChosen(k) = alpha;
   recoveredA(:,k) = aDelta;
   parameterHistory{k} = alphaInfo;
   alphaBracketFound(k) = alphaInfo.bracket_found;
   [phiDelta, ~] = data_misfit_only(aDelta, uDelta, f, h, w);
   residualCenter = sqrt(max(phiDelta,0));
   residualCenterAll(k) = residualCenter;
   residualRelative(k) = residualCenter / normUTrue;
   residualTargetRatio(k) = residualCenter / max(targetResidual, realmin);
   omegaCenter = aDelta.' * Q * aDelta;
   omegaCenterAll(k) = omegaCenter;
   [phiTrue, ~] = data_misfit_only(aTrue, uDelta, f, h, w);
   residualTrue = sqrt(max(phiTrue,0));
   resRatioTrue(k) = residualTrue / max(residualCenter, realmin);
   omegaRatioTrue(k) = omegaTrue / max(omegaCenter, realmin);

   % 相对误差以精确系数的离散 L2 范数归一化.
   trueRelativeError(k) = l2norm_discrete(aDelta-aTrue, w) / normATrue;
   fprintf(['delta=%.3f | alpha=%10.3e | 实际噪声=%7.4f | ','残差/目标=%6.3f | 真实相对误差=%8.4f | ', ...
            'C_res^ref=%7.4f | C_omega^ref=%7.4f | 是否夹住目标=%d\n'], ...
            delta, alpha, actualNoise(k), residualTargetRatio(k), ...
            trueRelativeError(k), resRatioTrue(k), omegaRatioTrue(k),alphaBracketFound(k));
   if ~alphaBracketFound(k)
       warning('delta=%.3f：alpha 搜索网格未夹住偏差原则目标，已采用最接近目标残差的结果。', delta);
   elseif abs(residualTargetRatio(k)-1) > 0.08
       warning('delta=%.3f：残差/目标残差=%.3f，请检查 alphaGrid。', delta, residualTargetRatio(k));
   end
   initialA = aDelta;
end
%% ------------------------ 构造四种系数规则 --------------------------
% 使真解进入后验可行集合所需的两个参考比值: 
%   C_res^ref   = ||F(a_true)-u_delta|| / ||F(a_delta)-u_delta||,
%   C_omega^ref = Omega(a_true) / Omega(a_delta).
% 所有选取的放大系数均向上保留两位小数

% 可容许单值参数: 每个噪声水平使用一个公共放大系数.
CsingleAdaptive = zeros(numberDelta,1);
for k = 1:numberDelta
   CsingleAdaptive(k) = ceil2(max([CsingleFloor, resRatioTrue(k), omegaRatioTrue(k)]));
end
% 可容许双值参数: 残差约束和稳定化约束分别选取放大系数.
CresAdaptive = zeros(numberDelta,1);
ComegaAdaptive = zeros(numberDelta,1);
for k = 1:numberDelta
   CresAdaptive(k) = ceil2(max(CdoubleFloor, resRatioTrue(k)));
   ComegaAdaptive(k) = ceil2(max(CdoubleFloor, omegaRatioTrue(k)));
end

% 固定参数仅在 delta=0.005 时校准一次, 随后保持不变.
deltaCalibration = 0.005;
calibrationIndex = find(abs(deltaList-deltaCalibration) < 100*eps(deltaCalibration), 1);
if isempty(calibrationIndex)
    error('deltaCalibration = %.3f 必须包含在 deltaList 中。', deltaCalibration);
end
CsingleFixed = CsingleAdaptive(calibrationIndex);
CresFixed = CresAdaptive(calibrationIndex);
ComegaFixed = ComegaAdaptive(calibrationIndex);

% 四列依次对应: 1 固定单值, 2 可容许单值, 3 固定双值, 4 可容许双值.
CresRule = zeros(numberDelta,4);
ComegaRule = zeros(numberDelta,4);
CresRule(:,1) = CsingleFixed;
ComegaRule(:,1) = CsingleFixed;
CresRule(:,2) = CsingleAdaptive;
ComegaRule(:,2) = CsingleAdaptive;
CresRule(:,3) = CresFixed;
ComegaRule(:,3) = ComegaFixed;
CresRule(:,4) = CresAdaptive;
ComegaRule(:,4) = ComegaAdaptive;
ruleNames = {'固定单值参数','可容许单值参数','固定双值参数','可容许双值参数'};
fprintf('\n放大系数规则（全部向上保留两位小数）\n');
fprintf('固定单值参数：C_res = C_omega = %.2f\n', CsingleFixed);
fprintf('固定双值参数：C_res = %.2f, C_omega = %.2f\n', CresFixed, ComegaFixed);
fprintf('可容许单值参数随 delta 的取值：%s\n', mat2str(CsingleAdaptive.', 4));
fprintf('可容许 C_res 随 delta 的取值：%s\n', mat2str(CresAdaptive.', 4));
fprintf('可容许 C_omega 随 delta 的取值：%s\n', mat2str(ComegaAdaptive.', 4));
%% =====================================================================
% 第 2 阶段: 对四种参数规则分别求解全局后验极值问题.
% ======================================================================
posteriorRelativeEstimate = nan(numberDelta,4);
posteriorAbsoluteEstimate = nan(numberDelta,4);
posteriorExtremal = cell(numberDelta,4);
posteriorInfo = cell(numberDelta,4);
trueSolutionFeasible = false(numberDelta,4);

% 在相邻噪声水平之间保存各策略对应的延续搜索方向.
previousWorst = cell(1,4);
fprintf('\n第 2 阶段：全局后验极值计算\n');
for k = 1:numberDelta
   delta = deltaList(k);
   aDelta = recoveredA(:,k);
   uDelta = uDeltaAll(:,k);
   for s = 1:4
       Cres = CresRule(k,s);
       Comega = ComegaRule(k,s);
       % 检查真解是否属于当前后验可行集合, 仅用于数值诊断. 
       trueSolutionFeasible(k,s) = (resRatioTrue(k) <= Cres*(1+1e-10)) && ...
           (omegaRatioTrue(k) <= Comega*(1+1e-10));
       extraDirections = [];
       if ~isempty(previousWorst{s})
           extraDirections = [previousWorst{s}-aDelta,aDelta-previousWorst{s}];
       end
       caseTimer = tic;
       [epsilonPost, aWorst, pInfo] = posterior_error_estimate( ...
           aDelta, uDelta, f, h, w, Q, lowerA, upperA,Comega, Cres, numberPosteriorStarts, ...
           numberSensitivityDirections, numberRandomDirections, numberRayScanPoints, extraDirections);
       posteriorAbsoluteEstimate(k,s) = epsilonPost;
       posteriorRelativeEstimate(k,s) = epsilonPost / normATrue;
       posteriorExtremal{k,s} = aWorst;
       posteriorInfo{k,s} = pInfo;
       previousWorst{s} = aWorst;
       fprintf(['delta=%.3f | %-15s | C_res=%.2f C_omega=%.2f | ', ...
                '真解可行=%d | 后验相对估计=%8.4f | 真实相对误差=%8.4f | ', ...
                'SQP 接受=%d/%d | 时间=%.1f 秒\n'], ...
                delta, ruleNames{s}, Cres, Comega,trueSolutionFeasible(k,s), ...
                posteriorRelativeEstimate(k,s), trueRelativeError(k), ...
                pInfo.number_accepted, numel(pInfo.exitflags), toc(caseTimer));
       if trueSolutionFeasible(k,s) && posteriorRelativeEstimate(k,s) + 1e-8 < trueRelativeError(k)
           warning('delta=%.3f，%s：真解可行，但当前求得的后验极值小于真实误差，请增大多初值/搜索设置。', delta, ruleNames{s});
       end
   end
end
%% ------------------------------ 汇总表 --------------------------------
summaryTable = table(deltaList(:), actualNoise, alphaChosen, trueRelativeError, ...
   resRatioTrue, omegaRatioTrue, ...
   CresRule(:,1), ComegaRule(:,1), posteriorRelativeEstimate(:,1), trueSolutionFeasible(:,1), ...
   CresRule(:,2), ComegaRule(:,2), posteriorRelativeEstimate(:,2), trueSolutionFeasible(:,2), ...
   CresRule(:,3), ComegaRule(:,3), posteriorRelativeEstimate(:,3), trueSolutionFeasible(:,3), ...
   CresRule(:,4), ComegaRule(:,4), posteriorRelativeEstimate(:,4), trueSolutionFeasible(:,4), ...
   'VariableNames', {'delta','actualNoise','alpha','trueRelError','CresRef','ComegaRef', ...
   'CresFixSingle','ComegaFixSingle','PostFixSingle','FeasFixSingle', ...
   'CresAdSingle','ComegaAdSingle','PostAdSingle','FeasAdSingle', ...
   'CresFixDouble','ComegaFixDouble','PostFixDouble','FeasFixDouble', ...
   'CresAdDouble','ComegaAdDouble','PostAdDouble','FeasAdDouble'});
disp(' ');
disp(summaryTable);
%% ------------------------------- 绘图 ----------------------------------
% 所有单个噪声水平的示意图统一取 delta=0.020.
deltaShow = 0.020;
kShow = find(abs(deltaList-deltaShow) < 100*eps(deltaShow), 1);
if isempty(kShow)
    error('用于展示的 delta=%.3f 不在 deltaList 中。', deltaShow);
end

% 图 1: delta=0.020 时的精确状态与含噪观测数据.
figure('Color','w','Name','delta=0.020 时的含噪观测数据');
plot(x, uTrue, 'k-','LineWidth',1.8);
hold on;
plot(x, uDeltaAll(:,kShow), 'ro-','LineWidth',1.1,'MarkerSize',4);
grid on;
xlabel('x');
ylabel('u(x)');
title(sprintf('含噪观测数据（\\delta = %.3f）', deltaShow));
legend('精确状态 u^{†}','含噪数据 u^{\delta}','Interpreter','tex','Location','best');
drawnow;

% 图 2: delta=0.020 时的精确系数与 Tikhonov 重构系数.
figure('Color','w','Name','delta=0.020 时的系数重构');
plot(x, aTrue, 'k-','LineWidth',1.8);
hold on;
plot(x, recoveredA(:,kShow), 'r--','LineWidth',1.6);
grid on;
xlabel('x');
ylabel('a(x)');
title(sprintf('系数重构结果（\\delta = %.3f）', deltaShow));
legend('精确系数 a^{†}','Tikhonov 重构系数 a^{\alpha,\delta}','Interpreter','tex','Location','best');
drawnow;

% 图 3: 真实相对误差与四种放大系数策略下的相对全局后验误差估计.
figure('Color','w','Name','全局后验误差估计');
plot(deltaList, trueRelativeError, 'ko-','LineWidth',1.8,'MarkerFaceColor','w','MarkerSize',7);
hold on;
plot(deltaList, posteriorRelativeEstimate(:,1), 'rs-','LineWidth',1.5,'MarkerSize',7);
plot(deltaList, posteriorRelativeEstimate(:,2), 'md-','LineWidth',1.5,'MarkerSize',7);
plot(deltaList, posteriorRelativeEstimate(:,3), 'b^-','LineWidth',1.5,'MarkerSize',7);
plot(deltaList, posteriorRelativeEstimate(:,4), 'gv-','LineWidth',1.5,'MarkerSize',7);
% 对固定参数已经不能包含真解的位置用叉号标记。
for s = [1,3]
    bad = ~trueSolutionFeasible(:,s);
    if any(bad)
        plot(deltaList(bad), posteriorRelativeEstimate(bad,s), 'kx','LineWidth',2.0,'MarkerSize',10);
    end
end
grid on;
xlabel('相对噪声水平 \delta');
ylabel('相对误差 / 相对后验误差估计');
title('不同噪声水平下的全局后验误差估计');
legend('Tikhonov 近似解真实相对误差','固定单值参数', '可容许单值参数', ...
       '固定双值参数', '可容许双值参数','Location','northwest');
drawnow;

% 图 4: 放大系数图, 加入 C=1.01 与 C=2.51 两条参考虚线. 
figure('Color','w','Name','后验可行集合放大系数');
plot(deltaList, CsingleAdaptive, 'ko-','LineWidth',1.6,'MarkerSize',6);
hold on;
plot(deltaList, resRatioTrue, 'bs-','LineWidth',1.5,'MarkerSize',6);
plot(deltaList, omegaRatioTrue, 'rd-','LineWidth',1.5,'MarkerSize',6);
yline(1.01, 'k--','LineWidth',1.2);
yline(2.51, 'k--','LineWidth',1.2);
grid on;
xlabel('相对噪声水平 \delta');
ylabel('可容许放大系数');
title('不同噪声水平下放大系数的选取');
legend('C_{min}', 'C_{res}^{ref}', 'C_{\Omega}^{ref}','参考线 C=1.01', '参考线 C=2.51', 'Location','best');
drawnow;

% 图 5: delta=0.020 时四种参数规则对应的后验极值系数.
figure('Color','w','Name','delta=0.020 时的后验极值系数');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for s = 1:4
    nexttile;
    plot(x, aTrue, 'k-','LineWidth',1.8);
    hold on;
    plot(x, recoveredA(:,kShow), 'r--','LineWidth',1.5);
    plot(x, posteriorExtremal{kShow,s}, 'b:','LineWidth',1.5);
    grid on;
    xlabel('x');
    ylabel('a(x)');
    title(sprintf('%s：C_{res}=%.2f，C_{\\Omega}=%.2f',ruleNames{s}, CresRule(kShow,s), ComegaRule(kShow,s)));
    legend('精确系数','Tikhonov 重构系数','后验极值系数','Location','best');
end
drawnow;

fprintf('\n===============================================================\n');
fprintf('总运行时间：%.1f 秒\n', toc(totalTimer));
fprintf('固定系数仅使用 delta=0.005 进行校准。\n');
fprintf('可容许系数在每个噪声水平下分别重新计算。\n');
fprintf('所有放大系数均向上保留两位小数。\n');
fprintf('单噪声水平示意图统一显示 delta=0.020。\n');
fprintf('===============================================================\n');
%% =======================广义偏差原则选择正则化参数======================
function [aBest, alphaBest, info] = choose_alpha_by_discrepancy( ...
   aInitial, uDelta, f, h, w, Q, lowerA, upperA, targetResidual,alphaGrid, numberRefinements)
   
   % 沿 alpha 从大到小做延拓，用对数二分逼近偏差原则方程
   numberGrid = numel(alphaGrid);    % 粗搜索网格中的参数个数
   recordsAlpha = nan(numberGrid + numberRefinements, 1);
   recordsResidual = nan(numberGrid + numberRefinements, 1);
   recordsExitflag = nan(numberGrid + numberRefinements, 1);
   recordsIterations = nan(numberGrid + numberRefinements, 1);
   recordCount = 0;                  % 当前保存求解结果次数
   bestDifference = inf;
   aBest = aInitial;                 % 若后续求解全部失败，返回初始系数
   alphaBest = alphaGrid(1);         % 默认参数设为网格中的第一个值
   previousAlpha = [];               % 保存上一个网格点的信息
   previousResidual = [];            % 用于判断当前残差是否跨过偏差目标
   previousA = [];
   alphaHigh = [];                   % 预先定义偏差原则目标两侧的信息
   alphaLow = [];
   aHigh = [];
   aLow = [];
   residualHigh = [];
   residualLow = [];
   aStart = aInitial;                % 设置 Tikhonov 优化的初值
   for j = 1:numberGrid
       alpha = alphaGrid(j);
       [aCurrent, residualCurrent, exitflag, output] = solve_tikhonov( ...
           aStart, alpha, uDelta, f, h, w, Q, lowerA, upperA);
       recordCount = recordCount + 1;
       recordsAlpha(recordCount) = alpha;
       recordsResidual(recordCount) = residualCurrent;
       recordsExitflag(recordCount) = exitflag;
       recordsIterations(recordCount) = output.iterations;

       % 衡量当前残差与目标残差的乘法相对距离
       difference = abs(log(max(residualCurrent, realmin) / targetResidual));
       if difference < bestDifference
           bestDifference = difference;
           aBest = aCurrent;
           alphaBest = alpha;
       end
       if ~isempty(previousAlpha) && previousResidual > targetResidual ...
               && residualCurrent <= targetResidual
           alphaHigh = previousAlpha;
           residualHigh = previousResidual;
           aHigh = previousA;
           alphaLow = alpha;
           residualLow = residualCurrent;
           aLow = aCurrent;
           break;
       end
       previousAlpha = alpha;
       previousResidual = residualCurrent;
       previousA = aCurrent;
       aStart = aCurrent;
   end

   % 若网格搜索没有夹住目标残差, 就返回最接近的结果.
   if isempty(alphaHigh)
       % 保存搜索历史, 并把 bracketFound 标记为 false.
       info = make_alpha_info(recordsAlpha, recordsResidual, recordsExitflag, ...
           recordsIterations, recordCount, targetResidual, false);
       return;
   end
   for j = 1:numberRefinements    % 对数二分细化
       alphaMiddle = sqrt(alphaHigh * alphaLow);
       if abs(log(alphaMiddle/alphaHigh)) <= abs(log(alphaMiddle/alphaLow))
           aStart = aHigh;
       else
           aStart = aLow;
       end
       [aMiddle, residualMiddle, exitflag, output] = solve_tikhonov( ...
           aStart, alphaMiddle, uDelta, f, h, w, Q, lowerA, upperA);
       recordCount = recordCount + 1;
       recordsAlpha(recordCount) = alphaMiddle;
       recordsResidual(recordCount) = residualMiddle;
       recordsExitflag(recordCount) = exitflag;
       recordsIterations(recordCount) = output.iterations;
       difference = abs(log(max(residualMiddle, realmin) / targetResidual));
       if difference < bestDifference
           bestDifference = difference;
           aBest = aMiddle;
           alphaBest = alphaMiddle;
       end
       if residualMiddle > targetResidual
           alphaHigh = alphaMiddle;
           residualHigh = residualMiddle;
           aHigh = aMiddle;
       else
           alphaLow = alphaMiddle;
           residualLow = residualMiddle;
           aLow = aMiddle;
       end
   end

   % 保存搜索历史, 并把 bracketFound 标记为 true.
   info = make_alpha_info(recordsAlpha, recordsResidual, recordsExitflag, ...
       recordsIterations, recordCount, targetResidual, true);
   info.alpha_bracket = [alphaLow, alphaHigh];
   info.residual_bracket = [residualLow, residualHigh];
end
%% =====================固定 α 求解 Tikhonov 问题=========================
function [aOpt, residual, exitflag, output] = solve_tikhonov(aStart, alpha, uDelta, f, h, w, Q, lowerA, upperA)
   
   aScale = 0.05;         % 设置变量尺度
   y0 = aStart / aScale;
   lb = (lowerA/aScale) * ones(size(aStart));           % 把系数上下界转换到缩放空间
   ub = (upperA/aScale) * ones(size(aStart));
   normalization = max(sum(w .* uDelta.^2), realmin);   % 计算目标函数归一化因子
   objective = @(y) tikhonov_objective_scaled(y, aScale, alpha, uDelta, f, h, w, Q, normalization);
   options = optimoptions('fmincon','Algorithm', 'sqp','Display', 'off','SpecifyObjectiveGradient', true, ...
       'MaxIterations', 900, 'MaxFunctionEvaluations', 120000,'OptimalityTolerance', 1e-9, ...
       'StepTolerance', 1e-12,'ConstraintTolerance', 1e-10,'TypicalX', ones(size(y0)));
   try
       [yOpt, ~, exitflag, output] = fmincon(objective, y0, [], [], [], [], lb, ub, [], options);
   catch optimizationError
       warning('Tikhonov 子问题求解失败');
       yOpt = min(max(y0, lb), ub);
       exitflag = -999;
       output = struct('iterations', 0, 'message', optimizationError.message);
   end
   aOpt = aScale * yOpt;            % 将缩放变量恢复成真实系数
   [phi, ~] = data_misfit_only(aOpt, uDelta, f, h, w);
   residual = sqrt(max(phi, 0));    % 计算最终系数的数据残差
end
%% ====================Tikhonov 目标函数和梯度============================
function [value, gradient] = tikhonov_objective_scaled(y, aScale, alpha, uDelta, f, h, w, Q, normalization)
   a = aScale * y;                  % 把优化变量 y 转换回系数 a
   [phi, gradPhi, ~] = data_misfit_and_gradient(a, uDelta, f, h, w);
                                    % 计算残差以及梯度
   omega = a.' * Q * a;             % 计算正则化泛函
   value = (phi + alpha * omega) / normalization;    % 返回归一化 Tikhonov 目标
   gradientA = gradPhi + 2*alpha*(Q*a);              % 计算目标关于 a 的梯度
   gradient = aScale * gradientA / normalization;    % 利用链式法则计算关于 y 的梯度
end
%% =========================后验误差估计计算==============================
function [epsilonPost, aWorst, info] = posterior_error_estimate(aCenter, uDelta, f, h, w, Q, lowerA, upperA, ...
   C_omega, C_data, numberStarts, numberSensitivityDirections,numberRandomDirections, numberRayScanPoints, extraDirections)
   
   % 通过多类方向构造可行初值，再用 SQP 求有限维后验极值问题
   % 搜索方向包括: 1) 低频余弦方向; 2) 局部正演 Jacobian 的弱敏感方向; 
   %   3) 固定的平滑随机方向; 4) 前一噪声水平得到的后验极值方向.
   [phiCenter, ~] = data_misfit_only(aCenter, uDelta, f, h, w);
   residualCenter = sqrt(max(phiCenter, 0));
   omegaCenter = aCenter.' * Q * aCenter;
   R2 = C_omega * omegaCenter;
   Delta2 = (C_data * residualCenter)^2;
   n = numel(aCenter);
   x = linspace(0, 1, n).';   % 重新生成网格, 用于构造余弦方向
   D = [];                    % 初始化方向矩阵, 每一列将存储一个搜索方向
   
   % 低频确定性方向
   for mode = 0:12
       if mode == 0
           direction = ones(n, 1);
       else
           direction = cos(mode*pi*x);
       end
       D = append_direction_pair(D, direction, w);
   end
   
   % Jacobian 弱敏感方向：这些方向对数据影响小，最可能形成较大的后验误差
   try
       Dsensitivity = weak_sensitivity_directions(aCenter, f, h, w, Q, numberSensitivityDirections);
       D = [D, Dsensitivity]; %#ok<AGROW>
   catch sensitivityError
       warning('弱敏感方向构造失败');
   end

   % 固定平滑随机方向，使不同噪声水平采用相同方向
   posteriorStream = RandStream('mt19937ar', 'Seed', 13579);
   smoother = speye(n) + 1e-5 * Q;    % 构造平滑矩阵
   for j = 1:numberRandomDirections
       direction = smoother \ randn(posteriorStream, n, 1);
       D = append_direction_pair(D, direction, w);
   end

   % 延续前一噪声水平已经找到的方向
   if nargin >= 15 && ~isempty(extraDirections)
       for j = 1:size(extraDirections, 2)
           D = append_direction_pair(D, extraDirections(:, j), w);
       end
   end
   numberDirections = size(D, 2);                  % 方向总数
   candidateSteps = zeros(numberDirections, 1);    % 存储每个方向上的最大可行步长
   candidatePoints = zeros(n, numberDirections);   % 存储对应候选点
   validCandidate = false(numberDirections, 1);    % 记录候选点是否确实满足后验约束
   for j = 1:numberDirections
       direction = D(:, j);

       % 沿射线搜索较远的可行步长
       step = largest_feasible_ray_step_scan( ...
           aCenter, direction, lowerA, upperA, uDelta, f, h, w, Q,R2, Delta2, numberRayScanPoints);
       if step > 1e-14    % 只有非平凡步长才生成候选点
           candidateSteps(j) = step;
           candidatePoints(:, j) = aCenter + 0.995*step*direction;  % 候选点取在估计边界的 99.5% 处
           validCandidate(j) = is_posterior_feasible(candidatePoints(:, j), uDelta, f, h, w, Q, R2, Delta2);
       end
   end
   candidateSteps = candidateSteps(validCandidate);     % 删除无效候选
   candidatePoints = candidatePoints(:, validCandidate);
   if isempty(candidateSteps)    % 无候选点时返回
       epsilonPost = 0;
       aWorst = aCenter;
       info = struct('exitflags', [], 'distances', 0,'R2', R2, 'Delta2', Delta2, ...
                     'number_accepted', 0, 'best_candidate_distance', 0, ...
                     'best_omega_ratio', 1/C_omega,'best_data_ratio', 1/C_data^2, ...
                     'message', '未找到非平凡可行射线。');
       return;
   end
   candidateDistances = sqrt(sum(w .* (candidatePoints-aCenter).^2, 1));% 计算每个候选点到中心的离散 L2 距离
   [bestDistance, bestCandidateIndex] = max(candidateDistances);        % 找到射线候选中最大的距离及其列编号
   aWorst = candidatePoints(:, bestCandidateIndex);                     % 把最远射线候选作为当前最坏点
   [~, order] = sort(candidateDistances, 'descend'); % 按距离从大到小排列候选点
   numberStarts = min(numberStarts, numel(order));   % 若有效候选少于计划，就只使用现有候选数
   order = order(1:numberStarts);                    % 选择最远的若干候选点作为 SQP 初值

   % 变量缩放与目标归一化.
   aScale = 0.05;
   lb = (lowerA/aScale) * ones(n, 1);
   ub = (upperA/aScale) * ones(n, 1);
   distanceScale2 = max(sum(w .* aCenter.^2), 1e-16);% 用中心点 L2 范数平方归一化后验目标
   options = optimoptions('fmincon','Algorithm', 'sqp','Display', 'off','SpecifyObjectiveGradient', true, ...
       'SpecifyConstraintGradient', true,'MaxIterations', 1200,'MaxFunctionEvaluations', 180000, ...
       'OptimalityTolerance', 1e-10,'StepTolerance', 1e-13,'ConstraintTolerance', 1e-8, ...
       'TypicalX', ones(n, 1));
   exitflags = zeros(numberStarts, 1);        % 每个初值的退出状态
   distances = zeros(numberStarts, 1);        % 每个 SQP 终点到中心的距离
   constraintValues = zeros(numberStarts, 2); % 记录结果是否被接受为可行结果
   accepted = false(numberStarts, 1);
   for j = 1:numberStarts
       y0 = candidatePoints(:, order(j)) / aScale;

       % 定义非线性约束函数
       objective = @(y) posterior_objective_scaled(y, aScale, aCenter, w, distanceScale2);
       nonlcon = @(y) posterior_constraints_scaled(y, aScale, uDelta, f, h, w, Q, R2, Delta2);
       try
           [yOpt, ~, exitflag] = fmincon(objective, y0, [], [], [], [], lb, ub, nonlcon, options);
       catch optimizationError
           warning('第 %d 个后验 SQP 初值求解失败：%s',j, optimizationError.message);
           yOpt = y0;
           exitflag = -999;
       end
       aOpt = aScale * yOpt;
       distance = l2norm_discrete(aOpt-aCenter, w);
       [c, ~] = posterior_constraints_unscaled(aOpt, uDelta, f, h, w, Q, R2, Delta2);
       exitflags(j) = exitflag;
       distances(j) = distance;
       constraintValues(j, :) = c(:).';
       accepted(j) = all(isfinite(aOpt)) && all(c <= 2e-6);
       if accepted(j) && distance > bestDistance
           bestDistance = distance;
           aWorst = aOpt;
       end
   end
   [phiWorst, ~] = data_misfit_only(aWorst, uDelta, f, h, w);
   omegaWorst = aWorst.' * Q * aWorst;
   epsilonPost = bestDistance;
   info = struct('exitflags', exitflags, 'distances', distances, ...
                 'constraint_values', constraintValues, ...
                 'accepted', accepted, ...
                 'number_accepted', nnz(accepted), ...
                 'candidate_distances', candidateDistances(:), ...
                 'best_candidate_distance', max(candidateDistances), ...
                 'number_directions', numberDirections, ...
                 'number_candidates', size(candidatePoints, 2), ...
                 'R2', R2, 'Delta2', Delta2, ...
                 'omega_center', omegaCenter, ...
                 'residual_center', residualCenter, ...
                 'best_omega_ratio', omegaWorst/R2, ...
                 'best_data_ratio', phiWorst/Delta2);
end
%% ========================定义非线性约束函数=============================
function D = append_direction_pair(D, direction, w)
   if isempty(direction) || any(~isfinite(direction))
       return;
   end
   nrm = l2norm_discrete(direction, w);  % 计算方向的离散 L2 范数
   if nrm <= 1e-12
       return;
   end
   direction = direction / nrm;          % 归一化
   D = [D, direction, -direction]; %#ok<AGROW>
end
%% ======================构造 Jacobian 弱敏感方向=========================
function D = weak_sensitivity_directions(a, f, h, w, Q, numberDirections)
% 构造局部正演 Jacobian 在 W_2^2 度量下的弱敏感方向, 在给定正则化能量下, 对观测数据的影响最小
   n = numel(a);
   J = forward_jacobian(a, f, h);
   Qsym = 0.5*(Q+Q.');
   Qsym = Qsym + 1e-12*speye(n);
   R = chol(Qsym);                    % 进行 Cholesky 分解: Q = R'*R
   Wsqrt = spdiags(sqrt(w), 0, n, n);
   B = full(Wsqrt * J / R);
   [~, ~, V] = svd(B, 'econ');
   numberDirections = min(numberDirections, size(V, 2));
   indices = size(V, 2)-numberDirections+1:size(V, 2);  % 选取 V 的最后若干列, 即最小奇异值对应的方向
   D = [];
   for index = indices
       direction = R \ V(:, index);
       D = append_direction_pair(D, direction, w);
   end
end
%% ==========================计算正演 Jacobian============================
function J = forward_jacobian(a, f, h)
   % 计算离散正演映射 F_n(a) 对 a 的 Jacobian
   % 先构造全部参数方向的右端项，再用一次 K\RHS 多右端求解
   n = numel(a);
   m = n-2;                       % 内部未知状态节点数
   K = diffusion_matrix(a, h);
   uInterior = K \ f(2:n-1);
   rightHandSides = zeros(m, n);
   for parameterIndex = 1:n
       dK = diffusion_matrix_derivative(parameterIndex, n, h);
       rightHandSides(:, parameterIndex) = -dK*uInterior;
   end
   J = zeros(n, n);
   J(2:n-1, :) = K \ rightHandSides;
end
%% ============构造 K(a) 对第 parameterIndex 个系数节点值的导数===========
function dK = diffusion_matrix_derivative(parameterIndex, n, h)
   m = n-2;
   daHalf = zeros(n-1, 1);
   if parameterIndex > 1    % 若 aj 不是第一个节点, 会影响左侧半节点
       daHalf(parameterIndex-1) = daHalf(parameterIndex-1) + 0.5;
   end
   if parameterIndex < n    % 若 aj 不是最后一个节点, 会影响右侧半节点
       daHalf(parameterIndex) = daHalf(parameterIndex) + 0.5;
   end

   % 根据正演矩阵主对角元计算其对 aj 的导数
   mainDiagonal = (daHalf(1:end-1) + daHalf(2:end)) / h^2;

   % 根据相邻非对角元根据相邻非对角元计算导数
   offDiagonal = -daHalf(2:end-1) / h^2;
   dK = sparse(1:m, 1:m, mainDiagonal, m, m);
   if m > 1  % 若矩阵维数大于 1，则加入上、下副对角线，得到完整对称三对角矩阵
       dK = dK + sparse(1:m-1, 2:m, offDiagonal, m, m) + sparse(2:m, 1:m-1, offDiagonal, m, m);
   end
end
%% ====================沿射线寻找最远可行步长============================
function step = largest_feasible_ray_step_scan(aCenter, direction,lowerA, upperA, uDelta, f, h, w, Q,R2, Delta2, numberScanPoints)
   % 沿射线先扫描再局部二分, 不再假设可行性只发生一次变化.
   positive = direction > 0;  % 分别找出方向中的正分量和负分量
   negative = direction < 0;
   stepBound = inf;
   if any(positive)  % ai ​+ tdi​ ≤ upperA
       stepBound = min(stepBound,min((upperA-aCenter(positive))./ direction(positive)));
   end
   if any(negative)  % ai​ + tdi​ ≥ lowerA
       stepBound = min(stepBound,min((lowerA-aCenter(negative)) ./ direction(negative)));
   end
   if ~isfinite(stepBound) || stepBound <= 0
       step = 0;
       return;
   end
   stepBound = 0.999 * stepBound;
   numberScanPoints = max(numberScanPoints, 21);  % 至少使用 21 个扫描点
   scanParameter = linspace(0, 1, numberScanPoints).^2;
   tGrid = stepBound * scanParameter;  % 把无量纲扫描参数转换成真实射线步长
   feasible = false(size(tGrid));      % 初始化每个扫描点的可行性
   feasible(1) = true;                 % aCenter 本身严格可行，无需求正演
   for j = 2:numel(tGrid)
       feasible(j) = is_posterior_feasible(aCenter+tGrid(j)*direction, uDelta, f, h, w, Q, R2, Delta2);
   end
   lastFeasible = find(feasible, 1, 'last');  % 找到扫描网格中最后一个可行点位置
   if isempty(lastFeasible) || lastFeasible == 1
       step = 0;
       return;
   end

   % 若最后一个扫描点仍然可行, 则直接返回盒约束允许的最远扫描步长
   if lastFeasible == numel(tGrid)
       step = tGrid(end);
       return;
   end
   left = tGrid(lastFeasible);
   right = tGrid(lastFeasible+1);
   for iteration = 1:55
       middle = 0.5*(left+right);
       if is_posterior_feasible(aCenter+middle*direction,uDelta, f, h, w, Q, R2, Delta2)
           left = middle;
       else
           right = middle;
       end
   end
   step = left;
end
%% ==============检查便宜的 W_2^2 约束；通过后才求解正问题================
function yes = is_posterior_feasible(a, uDelta, f, h, w, Q, R2, Delta2)
   if any(~isfinite(a))
       yes = false;
       return;
   end
   omega = a.' * Q * a;
   if omega > R2*(1+1e-10)
       yes = false;  % 若已经违反, 没有必要进行昂贵的正问题求解
       return;
   end
   [phi, ~] = data_misfit_only(a, uDelta, f, h, w);
   yes = phi <= Delta2*(1+1e-10);
end
%% ===========================后验目标函数==============================
function [value, gradient] = posterior_objective_scaled(y, aScale, aCenter, w, distanceScale2)
   a = aScale * y;                       % 恢复真实系数
   difference = a - aCenter;             % 计算候选点与恢复解的差
   distance2 = sum(w .* difference.^2);  %计算离散 L2 距离平方
   % fmincon 做最小化, 因此使用负的距离平方
   value = -distance2 / distanceScale2;
   gradientA = -2 * w .* difference / distanceScale2;
   gradient = aScale * gradientA;
end
%% ========================后验非线性约束及梯度==========================
function [c, ceq, GC, GCeq] = posterior_constraints_scaled(y, aScale, uDelta, f, h, w, Q, R2, Delta2)
   a = aScale * y;     % 将优化变量转换为系数
   [c, ceq, GCa, GCeq] = posterior_constraints_unscaled(a, uDelta, f, h, w, Q, R2, Delta2);

   % 不等式约束，等式约束，约束关于 a 的梯度，约束关于 a 的梯度
   GC = aScale * GCa;  % 链式法则
end
%% =============================未缩放版本===============================
function [c, ceq, GC, GCeq] = posterior_constraints_unscaled(a, uDelta, f, h, w, Q, R2, Delta2)
   
   % 计算数据残差平方及其梯度
   [phi, gradPhi, ~] = data_misfit_and_gradient(a, uDelta, f, h, w);
   omega = a.' * Q * a;    % 计算平滑泛函

   % 用无量纲形式改善 fmincon 的数值尺度
   c = [omega/R2 - 1; phi/Delta2 - 1];   % 把两个约束写成 fmincon 要求的形式
   ceq = [];     % 没有非线性等式约束
   GC = [2*(Q*a)/R2, gradPhi/Delta2];    % 构造两个不等式约束梯度
   GCeq = [];    % 没有等式约束梯度
end
%% ===========计算 phi(a)=||F_n(a)-u_delta||_{L2}^2 及其伴随梯度==========
function [phi, gradient, u] = data_misfit_and_gradient(a, uDelta, f, h, w)
   n = numel(a);
   K = diffusion_matrix(a, h);    % 构造正演矩阵
   u = zeros(n, 1);               % 首尾值保持为 0, 对应 Dirichlet 边界条件
   u(2:n-1) = K \ f(2:n-1);       % 求内部节点状态
   residual = u - uDelta;         % 计算状态残差
   phi = sum(w .* residual.^2);   % 计算离散数据残差平方

   % K 为对称正定矩阵, 伴随方程 K*lambda=W*(u-u_delta)
   lambda = zeros(n, 1);          % 初始化伴随变量
   lambda(2:n-1) = K \ (w(2:n-1) .* residual(2:n-1));  % 求解伴随方程

   % K(a)=B' diag(a_{i+1/2}) B / h^2, 且 a_{i+1/2}=(a_i+a_{i+1})/2
   edgeProduct = diff(lambda) .* diff(u) / h^2;  % 计算每条网格边上的乘积
   gradient = zeros(n, 1);
   gradient(1) = -edgeProduct(1);     % 第一个系数节点只影响第一条半网格边，梯度只包含一个边贡献
   gradient(end) = -edgeProduct(end); % 最后一个系数节点只影响最后一条边
   gradient(2:n-1) = -(edgeProduct(1:end-1) + edgeProduct(2:end));
end
%% =====================仅计算数据残差平方================================
function [phi, u] = data_misfit_only(a, uDelta, f, h, w)
   % 仅计算正演状态和数据失配, 不求伴随变量
   n = numel(a);
   K = diffusion_matrix(a, h);
   u = zeros(n, 1);
   u(2:n-1) = K \ f(2:n-1);
   residual = u-uDelta;
   phi = sum(w.*residual.^2);
end
%% ======================构造扩散方程离散矩阵===========================
function K = diffusion_matrix(a, h)
   % 有限体积/守恒中心差分离散 -[a(x)u'(x)]'
   n = numel(a);
   m = n - 2;
   aHalf = 0.5 * (a(1:end-1) + a(2:end));    % 用算术平均定义半节点系数
   mainDiagonal = (aHalf(1:end-1) + aHalf(2:end)) / h^2;  % 构造内部矩阵主对角元
   offDiagonal = -aHalf(2:end-1) / h^2;      % 构造副对角元
   K = sparse(1:m, 1:m, mainDiagonal, m, m); % 建立主对角稀疏矩阵
   if m > 1
       % 加入上副对角线和下副对角线, 得到对称三对角矩阵
       K = K + sparse(1:m-1, 2:m, offDiagonal, m, m) + sparse(2:m, 1:m-1, offDiagonal, m, m);
   end
end
%% =============================离散 L2 范数==============================
function value = l2norm_discrete(v, w)
   value = sqrt(max(sum(w .* v.^2), 0));
end
%% ====================================================================
function info = make_alpha_info(alpha, residual, exitflag, iterations,count, targetResidual, bracketFound)
   info = struct();
   info.alpha = alpha(1:count);
   info.residual = residual(1:count);
   info.exitflag = exitflag(1:count);
   info.iterations = iterations(1:count);
   info.target_residual = targetResidual;
   info.bracket_found = bracketFound;
end
