function otimizarna();
% Funcao de treinamento de RNA utilizando Gradiente e novos conceitos em AG 
% Sintaxe AA5();
% Tendo a RNA:   1 entrada
%                5 neuronios tipo tansig na camada escondida
%                1 neuronio linear na saida
clc;
close all;
rand('seed',33);        % Garante repetibilidade do experimento
randn('seed',55);             
Precisao=1e-3;          % Precisao com que se deseja obter o ERRO (define o tamanho do cromossomo)
%============================================================================
% DEFINICAO DA REDE, DA FUNCAO A SER APRENDIDA E DO ERRO
Alfa=0.01;               % Passo de aprendizagem no treinamento pelo Gradiente
Ptrain=100;                % Numero de passos para o treinamento
npop=15;                   % Tamanho da populaÃ§Ã£o para implementacao dos AG
PopMax=2*npop;                 % Tamanho mÃ¡ximo da populaÃ§Ã£o a partir da inicial
%Rede neural:
%Yo=tansig(W1*X)        Saida da camada oculta (Yo e W1 -> vetores 1x5; X -> escalar)
%Yn=W2*Yo'              Saida estimada pela RNA (Yn -> escalar; W2 -> vetor 1x5)
%Funcao a aprender:  
%Y=X.^4-X.^3-3*X.^2
%e=Y-Yn        Erro cometido pela RNA, que no caso é utilizada como a função a ser otimizada
funcao="(X(i).^4-X(i).^3-3*X(i).^2) - (W2(1,:)*tansig(W1(1,:)*X(i))')"; % e = Y(i)-Yn(i)
grad='[-W2(1,:).*(1-(W1(1,:)*X(i)).^2)*X(i)      -tansig(W1(1,:)*X(i))]';
%============================================================================
% FASE 1: Criacao da populacao inicial
disp('FASE 1: Criacao e treinamento da populacao inicial.');
populacao=gerapop(npop); % Inicializa nova populacao
partida=populacao;             % Inicializa os pontos de partida com a populacao atual 
dE=zeros(1,npop);              % So para inicializar variavel
for i=1:npop       % Treinamento
    [individuo,Err,varcus,DistE]=train(funcao,grad,populacao(i,:),Ptrain,Alfa);
    populacao(i,:)=individuo;
    J(i)=Err;       %custo é o erro (Y-Yn)
    dJ(i)=varcus;   %variação do custo com o treinamento
    dE(i)=dE(i)+DistE; %distancia euclidiana acumulada
end                   
[populacao,partida,J]=ordem(populacao,partida,J);   % Coloca na ordem de J
plot(J)
title('Custo (em ordem) dos individuos da populacao inicial (fase 1)')
xlabel('Individuo')
ylabel('Erro')
disp('Pressione Enter para continuar')
pause
Ptrain=10*Ptrain;         % Garante épocas de treinamento para convergência no futuro 
%============================================================================
% FASE 2: Evolucao genetica - Varredura completa do espaco de busca
clc;
close all;
disp('FASE 2: Evolucao genetica - Varredura completa no espaco de busca.');

while (size(populacao,1)<PopMax)
    filho=gera_filho(partida);% Gera novo filho

    partida(size(partida,1)+1,:)=filho; % Guarda ponto de partida

    [filho,Err,varcus,DistE]=train(funcao,grad,filho,Ptrain,Alfa);  % Treinamento
    populacao(size(populacao,1)+1,:)=filho;% Insere individuo treinado na populacao
    J(size(populacao,1))=Err;           % Criterio de adaptabilidade = Erro
    dJ(size(populacao,1))=varcus;
    %dE(size(populacao,1))= dE(size(populacao,1)) + DistE;
        
    % Reorganiza populacao com novo individuo
    [populacao,partida,J]=ordem(populacao,partida,J); % Coloca na ordem de J

end
[populacao,partida,J]=ordem(populacao,partida,J); % Coloca na ordem de J
% Populacao no limite maximo permitido - executa eliminacao
if size(populacao,1)>=PopMax
   disp('Eliminacao de individuos');
   populacao=populacao(1:npop,:);     % Ajusta tamanho da populacao (menos 1)
   partida=partida(1:npop,:);         % Ajusta tamanho dos pontos de partida
   J=J(1:npop);
end
Ptrain=5*Ptrain;         % Garante épocas de treinamento para convergencia no futuro
plot(J)
title('Custo (em ordem) dos individuos da populacao evoluida (fase 2)')
xlabel('Individuo')
ylabel('Erro')
disp('Pressione Enter para continuar')
pause
%============================================================================
% FASE 3: Aperfeicoamento da populacao final
close all;
disp(' ');
disp('FASE 3: Aperfeicoamento do melhor individuo da populacao final.');
dJfim=1e5;       % So para inicializar variÃ¡veis
conta=20;                % Numero maximo de loops de treinamento (evita loop muito longo)
while dJfim>Precisao && conta>0    % Forca o treinamento ate alcancar precisao desejada
    for i=1:npop       % Treinamento
        [individuo,Err,varcus,DistE]=train(funcao,grad,populacao(i,:),Ptrain,Alfa/10);
        populacao(i,:)=individuo;
        dJfim=varcus;  % Verifica a variacao do indice J com o treinamento
        J(i)=Err;       %custo é o erro (Y-Yn)
        dE(i)=dE(i)+DistE;
    end
    conta=conta-1;        % Decrementa contador de loops de treinamento
end
[populacao,partida,J]=ordem(populacao,partida,J); % Coloca na ordem de J
W1=populacao(1,1:5);
W2=populacao(1,6:10);

plot(J)
title('Custo (em ordem) dos individuos da populacao final aperfeicoada (fase 3)')
xlabel('Individuo')
ylabel('Erro')
disp('Pressione Enter para continuar')
pause

disp(' ')
disp('O melhor individuo da populacao final eh o escolhido.')
disp('Este individuo eh composto pelos pesos:')
disp('W1:')
disp(W1)
disp('e W2:')
disp(W2)
disp('O respectivo erro deste individuo eh:')
disp(J(1))
%============================================================================
% FUNCOES UTEIS PARA O ALGORITMO
%============================================================================
function filho=gera_filho(populacao); % Gera novo filho
TamP=size(populacao,1);              % Tamanho da populacao (PAIS e MAES possiveis)
BestP=ceil(TamP*0.2);                % Garante um pai dentro dos 20% melhores 
nPai=ceil(rand*BestP);               % Seleciona pai dentro dos BestP melhores
nMae=nPai;                              % So para inicializar variaveis
while nMae==nPai               % Individuo pai diferente do individuo mae
    nMae=ceil(rand*TamP);       % Seleciona mae dentro de toda a populacao
end
pai=populacao(nPai,:);            % Dados do Pai na populacao
mae=populacao(nMae,:);            % Dados da Mae na populacao
filho=mean([pai;mae])+randn(size(pai))*sqrt(sumsqr(pai-mae))/2;  % Euristica para gerar o filho
% Probabilidade de gerar filho com super-mutacao 
if rand<0.30                      % Probabilidade de 30% de gerar super-mutante
    Fator=max(sqrt([sumsqr(pai) sumsqr(mae)  1])); % Maior distancia euclidiana (>=1)
    filho=randn*(pai-mae);         % Direcao e sentido para Super-Mutacao
    filho=Fator*filho+mean([pai;mae]); % Novo ponto de busca (distante)
end
%============================================================================
function [nind,J,dJ,DistE]=train(funcao,grad,individuo,Ptrain,passo);

% Define entradas/saidas para treinamento(poderiam ser dados adquiridos)
X=rand(Ptrain,1);        % Entradas para treinamento
Y=X.^4-X.^3-3*X.^2;                % Saida conhecida (TARGET - Alvo)

% Treinamento de um individuo
W1=individuo(1,1:5);
W2=individuo(1,6:10);
velhoind=individuo;                      % So para guardar o valor antes do treinameto
V1=[];
V2=[];
for i=1:Ptrain % para dar o valor aos indices da string em 'funcao'
    V1(i)=abs(eval(funcao));           % Guarda valor da funcao no ponto inicial
end
for i=1:Ptrain             % Numero de iteracoes para o treinamento   
   hk=-eval(grad);         % Direcao do gradiente negativo
   lambda=calc_min(funcao,hk,W1,W2,passo,X);   % Calcula o melhor passo nesta direcao
  
   W1=W1+lambda(1)*hk(1:5);          % Atualiza xk na direcao hk com o melhor passo possivel
   W2=W2+lambda(2)*hk(6:10);
   V=abs(eval(funcao));         % Novo valor da funcao
   if (V-min(V1))>0             % Funcao custo aumentou de valor
      clc
      disp('Custo diminuindo');
      break
   else
       disp('Custo aumentando, risco de nao convergir');
   end
end
for i=1:Ptrain % para dar o valor aos indices da string em 'funcao'
    V2(i)=abs(eval(funcao)); % Guarda valor da funcao no ponto final
    dJparcial= V1(i)-V2(i); % Variacao do custo com o treinamento
end            
J=min(V2);                      % Custo alcancado com o treinamento
dJ=min(dJparcial);             % Variacao do custo com o treinamento     
nind=[W1 W2];             % Atualiza novo individuo 
d=nind-velhoind;                    % Verifica distancia percorrida pelo individuo
DistE=sqrt(d*d');          % Distancia Euclidiana
%============================================================================================
function lambda=calc_min(funcao,hk,W1k,W2k,passo,X);
% Calculo de minimo unidimensional utilizando metodo de busca uniforme
tol=1e-6;             % Tolerancia para condicao de saida do loop de busca
if passo<tol          % So para evitar inconsistencias
   tol=passo/100;
end
W1=W1k;                 % Inicializa x de busca
W2=W2k;
parou=0;              % Verifica se algoritmo nao consegue a tolerancia desejada
while abs(passo)>tol & ~parou
   i=1; % para dar o valor de 1 aos indices da string em 'funcao'
   Jk=abs(eval(funcao));   % Valor da funcao no ponto x anterior
   W1=W1+passo*hk(1,1:5);      % Adianta 1 passo
   W2=W2+passo*hk(1,1:5);
   Jk1=abs(eval(funcao));  % Valor da funcao no novo ponto x
   delta= Jk1-Jk;     % Variacao no valor da funcao custo
   if delta > 0       % Funcao no sentido crescente, esta indo para o lado errado
      passo=-passo/2; % Inverte o sentido e diminui o passo
   end
   if delta<tol
      parou=1;
   end
end
lambda1=(W1-W1k)/hk(1,1:5);     % Calcula o valor de lambda para o ponto: x = xk + lambda*hk
lambda2=(W2-W2k)/hk(1,6:10);
lambda=[lambda1 lambda2];
%============================================================================
function [populacao]=gerapop(npop);
% Funcao para criar arquivo com nova populacao na regiao especificada
for i=1:npop                            % Cria cada elemento da populacao
    W1=0.1*randn(1,5);           % Dado aleatorio dentro da faixa especificada
    W2=0.1*randn(1,5);
    populacao(i,:)= [W1(1,:) W2(1,:)];               % Introduz novo individuo na populacao
end
%============================================================================
function [npopulacao,npartida,nJ]=ordem(populacao,partida,J);   % Coloca na ordem de J
[Y,I]=sort(J);          % Coloca em ordem de J e sai com indice I
for i=1:length(I)       % Varre o indice
   npopulacao(i,:)=populacao(I(i),:);
   npartida(i,:)=partida(I(i),:);
   nJ(i)=J(I(i));
end
%============================================================================