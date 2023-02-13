$$
\gdef\w{\boldsymbol{\varpi}}
\gdef\x{\boldsymbol{\xi}}
\gdef\e{\boldsymbol{\epsilon}}
\gdef\p{\boldsymbol{\rho}}
\gdef\J{\mathcal{J}}
\gdef\T{\mathcal{T}}
\gdef\g{\hat{\boldsymbol{\gamma}}_k}
$$

$$
\begin{align*}
\hat{\w} &= \J ( \ln(\hat{\mathbf{T}}(\tau) \hat{\mathbf{T}}_k^{-1})^\vee) (\boldsymbol{\Lambda}_2(\tau) \g(t_k) + \boldsymbol{\Psi}_2(\tau) \g(t_{k+1}) ) \\
&= \J(\boldsymbol{\Lambda}_1(\tau) \g(t_k) + \boldsymbol{\Psi}_1(\tau) \g(t_{k+1}) ) \Big[ \underbrace{\boldsymbol{\Lambda}_2(\tau) \g(t_k) + \boldsymbol{\Psi}_2(\tau) \g(t_{k+1})}_{\w_\tau} \Big]
\end{align*}
$$

$$
\g(t_k) = \begin{bmatrix} \mathbf{0} \\ \hat{\w}_k \end{bmatrix},\quad \g(t_{k+1}) = \begin{bmatrix} \ln(\hat{\mathbf{T}}_{k+1, k})^\vee \\ \J(\ln(\hat{\mathbf{T}}_{k+1, k})^\vee)^{-1} \hat{\w}_{k+1} \end{bmatrix}
$$

$$
\g(t_{k+1}) \approx \begin{bmatrix} \ln(\hat{\mathbf{T}}_{\text{op},k+1, k})^\vee + \J_{\text{op},k+1,k}^{-1}(\e_{k+1} - \T_{\text{op},k+1,k}\e_k) \\
\left( \J_{\text{op},k+1,k}^{-1} - \frac{1}{2}\left( \J_{\text{op},k+1,k}^{-1}(\e_{k+1} - \T_{\text{op},k+1, k} \e_k)   \right)^\curlywedge   \right) (\w_{\text{op},k+1} + \boldsymbol{\eta}_{k+1})
\end{bmatrix}
$$

$$
\begin{align*}
\frac{\partial \hat{\w}(\tau)}{\partial \mathbf{x}} &= \J_{\text{op},\tau,k} \frac{\partial}{\partial \mathbf{x}} \Big[ \boldsymbol{\Lambda}_2(\tau) \g(t_k) + \boldsymbol{\Psi}_2(\tau) \g(t_{k+1}) \Big] \\
&-\frac{1}{2} \w^\curlywedge_{\text{op},\tau}  \frac{\partial}{\partial \mathbf{x}} \Big[ \boldsymbol{\Lambda}_1(\tau) \g(t_k) + \boldsymbol{\Psi}_1(\tau) \g(t_{k+1}) \Big] 
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \w_\tau}{\partial \e_{k+1}} &= \Psi_{21} \J_{\text{op},k+1,k}^{-1}  + \frac{1}{2} \Psi_{22} \w_{\text{op},k+1}^\curlywedge \J_{\text{op},k+1,k}^{-1} \\
\frac{\partial \w_\tau}{\partial \e_k} &= -\Psi_{21} \J_{\text{op},k+1,k}^{-1} \T_{\text{op},k+1,k} - \frac{1}{2} \Psi_{22} \w_{\text{op},k+1}^\curlywedge \J_{\text{op},k+1,k}^{-1} \T_{\text{op},k+1,k} \\
\frac{\partial \w_\tau}{\partial \boldsymbol{\eta}_k} &= \Lambda_{22} \\
\frac{\partial \w_\tau}{\partial \boldsymbol{\eta}_{k+1}} &= \Psi_{22} \J_{\text{op},k+1,k}^{-1} \\
\end{align*}
$$
