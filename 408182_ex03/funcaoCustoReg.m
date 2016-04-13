function [J, grad] = funcaoCustoReg(theta, X, y, lambda)
%FUNCAOCUSTOREG Calcula o custo da regressao logistica com regularizacao
%   J = FUNCAOCUSTOREG(theta, X, y, lambda) calcula o custo de usar theta 
%   como parametros da regressao logistica para ajustar os dados de X e y 

% Initializa algumas variaveis uteis
m = length(y); % numero de exemplos de treinamento

% Voce precisa retornar as seguintes variaveis corretamente
J = 0;
grad = zeros(size(theta));

% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Calcule o custo de uma escolha particular de theta.
%             Voce precisa armazenar o valor do custo em J.
%             Calcule as derivadas parciais e encontre o valor do gradiente
%             para o custo com relacao ao parametro theta
%
% Obs: grad deve ter a mesma dimensao de theta
%

% g = sigmoid(X * theta);

% norm_theta = theta;
% norm_theta(1) = 0; % discards first theta value

%norm = (lambda / 2*m) * sum(norm_theta .^ 2);

%J = sum(y .* log(g) + (1 - y) .* log(1 - g)) / -m;
%J = J + norm;

%grad = ((g - y)' * X) / m;

% since normalization is applied from the cost function, get initial
% values:
[J, grad] = funcaoCusto(theta, X, y);  
% calc normalization and add it to initial cost and gradient values:
J = J + (lambda * sum(theta .^ 2)) / (2 * m);
grad = (grad' + (theta * lambda) / m);


% =============================================================

end
