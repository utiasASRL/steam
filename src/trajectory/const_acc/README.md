$$
\gdef\w{\boldsymbol{\varpi}}
\gdef\dw{\dot{\boldsymbol{\varpi}}}
\gdef\x{\boldsymbol{\xi}}
\gdef\e{\boldsymbol{\epsilon}}
\gdef\p{\boldsymbol{\rho}}
\gdef\J{\mathcal{J}}
\gdef\T{\mathcal{T}}
\gdef\g{\hat{\boldsymbol{\gamma}}_k}
$$

$$
\begin{align*}
\g(t_k) &= \begin{bmatrix} \mathbf{0} \\ \w_k \\ \dw_k \end{bmatrix} \\
\g(t_{k+1}) &= \begin{bmatrix} \ln(\mathbf{T}_{k+1,k})^\vee \\ \J_{k+1,k}^{-1} \w_{k+1} \\ -\frac{1}{2}(\J_{k+1,k}^{-1} \w_{k+1})^\curlywedge \w_{k+1} + \J_{k+1,k}^{-1} \dw_{k+1} \end{bmatrix} \\
\end{align*}
$$

$$
\g(t_{k+1})\approx \begin{bmatrix} \ln(\hat{\mathbf{T}}_{\text{op},k+1, k})^\vee + \J_{\text{op},k+1,k}^{-1}(\delta\x_{k+1} - \T_{\text{op},k+1,k}\delta\x_k) \\
\left( \J_{\text{op},k+1,k}^{-1} - \frac{1}{2}\left( \J_{\text{op},k+1,k}^{-1}(\delta\x_{k+1} - \T_{\text{op},k+1, k} \delta\x_k)   \right)^\curlywedge   \right) (\w_{\text{op},k+1} + \delta\w_{k+1}) \\
\mathbf{e}^\prime + \J_{\text{op},k+1,k}^{-1} \dw_{\text{op},k+1} + \frac{1}{2}\dw_{\text{op},k+1}\J_{k+1,k}^{-1} \delta\x_{k+1,k} + \J_{\text{op},k+1,k}^{-1} \delta \dw_{k+1}
\end{bmatrix}
$$

From Tim Tang's thesis:

$$
\begin{align*}
\mathbf{e}^\prime &= -\frac{1}{2} (\J_{k+1,k}^{-1} \w_{k+1})^\curlywedge \w_{k+1} \\
&\approx \overline{\mathbf{e}} -\frac{1}{2} \left( (\J_{\text{op},k+1,k}^{-1} \w_{\text{op},k+1})^\curlywedge \delta \w_{k+1} - \w_{\text{op},k+1}^\curlywedge \J_{\text{op},k+1,k}^{-1}\delta\w_{k+1} -\frac{1}{2}\w_{\text{op},k+1}^\curlywedge \w_{\text{op},k+1}^\curlywedge \J_{\text{op},k+1,k}^{-1} \delta\x_{k+1,k}
\right)
\end{align*}
$$

where $\delta\x_{k+1,k} = \J_{\text{op},k+1,k}^{-1}(\delta\x_{k+1} - \T_{\text{op},k+1,k}\delta\x_k)$.

The Jacobians for the prior cost terms are included in Tim Tang's thesis, these can be used to do covariance interpolation as well.

Interpolation Jacobians:

$$
\begin{align*}
&\mathbf{T}(\tau) = \exp \Big(\Big( \Lambda_{12} \w_{k} + \Lambda_{13} \dw_{k} + \Omega_{11} \ln (\mathbf{T}_{k+1,k})^\vee \\
&+\Omega_{12}\J_{k+1,k}^{-1}\w_{k+1} + \Omega_{13} \Big(-\frac{1}{2}(\J_{k+1,k}^{-1} \w_{k+1})^\curlywedge \w_{k+1} + \J_{k+1,k}^{-1} \dw_{k+1} \Big)\Big)^\wedge \Big) \mathbf{T}_{k}
\end{align*}
$$

$$
\frac{\partial \mathbf{T}(\tau)}{\partial \mathbf{x}} = \J_{\text{op},\tau} \frac{\partial \x_{\tau}}{\partial \mathbf{x}} + \T_{\text{op},\tau} \frac{\partial \mathbf{T}_k}{\partial \mathbf{x}}
$$

$$
\w(\tau) = \J(\x_{\tau}) \dot{\x}_{\tau}
$$

$$
\frac{\partial \w(\tau)}{\partial \mathbf{x}} = \J_{\text{op},\tau} \frac{\partial \dot{\x}_{\tau}}{\partial \mathbf{x}} - \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge  \frac{\partial \x_{\tau}}{\partial \mathbf{x}}
$$

$$
\dw(\tau) \approx \J(\x_{\tau}) (\ddot{\x}_{\tau} + \frac{1}{2} \dot{\x}_{\tau}^\curlywedge \w(\tau) )
$$

$$
\frac{\partial \dw(\tau)}{\partial \mathbf{x}} = \J_{\text{op},\tau} \frac{\partial}{\partial \mathbf{x}} \Big(\ddot{\x}_{\tau} + \frac{1}{2} \dot{\x}_{\tau}^\curlywedge \w(\tau) \Big) - \frac{1}{2} \Big( \ddot{\x}_{\text{op},\tau} + \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge \w_\text{op}(\tau) \Big)^\curlywedge \frac{\partial \x_{\tau}}{\partial \mathbf{x}}
$$

$$
\begin{align*}
&\frac{\partial}{\partial \mathbf{x}} \Big(\ddot{\x}_{\tau} + \frac{1}{2} \dot{\x}_{\tau}^\curlywedge \w(\tau) \Big) = \frac{\partial \ddot{\x}_{\tau}}{\partial \mathbf{x}} + \frac{1}{2} \frac{\partial}{\partial \mathbf{x}} \dot{\x}_{\tau}^\curlywedge \w(\tau) \\
 &= \frac{\partial \ddot{\x}_{\tau}}{\partial \mathbf{x}} - \frac{1}{2} \w_{\text{op}}(\tau)^\curlywedge \frac{\partial \dot{\x}_{\tau}}{\partial \mathbf{x}}  + \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge \frac{\partial \w(\tau)}{\partial \mathbf{x}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \dw(\tau)}{\partial \mathbf{x}} =~&\J_{\text{op},\tau} \Bigg(  \frac{\partial \ddot{\x}_{\tau}}{\partial \mathbf{x}} - \frac{1}{2} \w_{\text{op}}(\tau)^\curlywedge \frac{\partial \dot{\x}_{\tau}}{\partial \mathbf{x}}  + \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge \Big( \J_{\text{op},\tau} \frac{\partial \dot{\x}_{\tau}}{\partial \mathbf{x}} - \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge  \frac{\partial \x_{\tau}}{\partial \mathbf{x}} \Big)   \Bigg) \\
&- \frac{1}{2} \Big( \ddot{\x}_{\text{op},\tau} + \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge \w_\text{op}(\tau) \Big)^\curlywedge \frac{\partial \x_{\tau}}{\partial \mathbf{x}} \\
\frac{\partial \dw(\tau)}{\partial \mathbf{x}} =~&\J_{\text{op},\tau} \frac{\partial \ddot{\x}_{\tau}}{\partial \mathbf{x}} \\
&+ \J_{\text{op},\tau} \left( - \frac{1}{2} \w_{\text{op}}(\tau)^\curlywedge  + \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge \J_{\text{op},\tau} \right) \frac{\partial \dot{\x}_{\tau}}{\partial \mathbf{x}} \\
&+ \left(-\frac{1}{4} \J_{\text{op},\tau}  \dot{\x}_{\text{op},\tau}^\curlywedge \dot{\x}_{\text{op},\tau}^\curlywedge - \frac{1}{2} \Big( \ddot{\x}_{\text{op},\tau} + \frac{1}{2} \dot{\x}_{\text{op},\tau}^\curlywedge \w_\text{op}(\tau) \Big)^\curlywedge \right)  \frac{\partial \x_{\tau}}{\partial \mathbf{x}}
\end{align*}
$$

$$
\begin{align*}
\x_\tau = \x(\tau) =~&\boldsymbol{\Lambda}_{12}(\tau) \w_k + \boldsymbol{\Lambda}_{13}(\tau) \dot{\w}_k + \boldsymbol{\Omega}_{11}(\tau) \ln(\mathbf{T}_{k+1,k})^\vee + \boldsymbol{\Omega}_{12}(\tau) \J_{k+1,k}^{-1} \w_{k+1} \\
&+ \boldsymbol{\Omega}_{13} \left(-\frac{1}{2}\left(\J_{k+1,k}^{-1} \w_{k+1}\right)^\curlywedge \w_{k+1} + \J_{k+1,k}^{-1} \dw_{k+1} \right)
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \x_\tau}{\partial \delta\x_{k+1}} =~&\Omega_{11} \J_{\text{op},k+1,k}^{-1} + \frac{1}{2}\Omega_{12} \w_{\text{op},k+1}^\curlywedge\J_{\text{op},k+1,k}^{-1} \\
&+\frac{1}{4}\Omega_{13} \w_{\text{op},k+1}^\curlywedge\w_{\text{op},k+1}^\curlywedge \J_{\text{op},k+1,k}^{-1} + \frac{1}{2} \Omega_{13} \dw_{\text{op},k+1}^\curlywedge \J_{\text{op},k+1,k}^{-1} \\
\frac{\partial \x_\tau}{\partial \delta\x_{k}} =~&-\frac{\partial \x_\tau}{\partial \delta\x_{k+1}} \T_{\text{op},k+1,k} \\
\frac{\partial \x_\tau}{\partial \delta \w_{k}}=~&\Lambda_{12} \mathbf{1} \\
\frac{\partial \x_\tau}{\partial \delta \w_{k+1}} =~&\Omega_{12}\J_{\text{op},k+1,k}^{-1} -\frac{1}{2} \Omega_{13} \left(\left(\J_{\text{op},k+1,k}^{-1}\w_{\text{op},k+1}  \right)^\curlywedge - \w_{\text{op},k+1}^\curlywedge\J_{\text{op},k+1,k}^{-1}  \right) \\
\frac{\partial \x_\tau}{\partial \delta \dw_{k}} =~&\Lambda_{13} \mathbf{1} \\
\frac{\partial \x_\tau}{\partial \delta \dw_{k+1}} =~&\Omega_{13} \J_{\text{op},k+1,k}^{-1}
\end{align*}
$$

Note that the Jacobians of $\dot{\x}_\tau$ and $\ddot{\x}_\tau$ have the same form as above, except that we use the second and third row of the interpolation matrices respectively. For example, $\frac{\partial \dot{\x}_\tau}{\partial \delta\x_{k+1}} = \Omega_{21} \J_{\text{op},k+1,k}^{-1} + \frac{1}{2}\Omega_{22}\w_{\text{op},k+1}^\curlywedge\J_{\text{op},k+1,k}^{-1} + \cdots~.$

## Extrapolation

WNOA:

$$
\begin{align*}
\x_k(\tau) &= \ln(\mathbf{T}(t) \mathbf{T}_k^{-1})^\vee \\
\dot{\x}_k(\tau) &= \J(\x_k(\tau))^{-1} \w(t)
\end{align*}
$$

Past the end knot: $\x_k = \mathbf{0}^T$ and so $\J(\x_k(\tau))^{-1} = \mathbf{1}$, $\dot{\x}_k = \w_k$.

$$
\boldsymbol{\gamma}_\tau = \begin{bmatrix} \mathbf{1} & \Delta t \\ \mathbf{0} & \mathbf{1} \end{bmatrix} \begin{bmatrix} \x_k \\ \dot{\x}_k \end{bmatrix} = \begin{bmatrix} \Delta t \w_k \\ \w_k \end{bmatrix}
$$

WNOJ:

Past the end knot: $\x_k = \mathbf{0}^T$ and so $\J(\x_k(\tau))^{-1} = \mathbf{1}$, $\dot{\x}_k = \w_k$.

$$
\begin{align*}
\ddot{\x}_k &= -\frac{1}{2} \dot{\x}_k(t)^\curlywedge \w(t) + \J(\x_k(t))^{-1} \dw(t) \\
&= \dw_k
\end{align*}
$$

So the extrapolation for WNOJ looks like this:

$$
\boldsymbol{\gamma}_\tau = \begin{bmatrix} \x_\tau \\ \dot{\x}_\tau \\ \ddot{\x}_\tau \end{bmatrix} =  \underbrace{\begin{bmatrix} \mathbf{1} & \Delta t \mathbf{1} & \frac{1}{2} \Delta t^2 \mathbf{1} \\ \mathbf{0} & \mathbf{1} & \Delta t \mathbf{1} \\ \mathbf{0} & \mathbf{0} & \mathbf{1} \end{bmatrix}}_{\boldsymbol{\Phi}(\tau, t_k)} \underbrace{\begin{bmatrix} \x_k \\ \dot{\x}_k \\ \ddot{\x}_k \end{bmatrix}}_{\boldsymbol{\gamma}_k(t_k)} = \begin{bmatrix} \Delta t \w_k + \frac{1}{2}\Delta t^2 \dw_k \\ \w_k + \Delta t \dw_k \\ \dw_k \end{bmatrix}
$$

We can use this formula for doing the extrapolation of any GP prior including WNOA, WNOJ, Singer, etc:

$$
\boldsymbol{\gamma}(\tau) = \boldsymbol{\Phi}(\tau, t_k) \boldsymbol{\gamma}_k(t_k)
$$

In order to convert from local to global variables, we do the following:

$$
\begin{align*}
\mathbf{T}(\tau) &= \exp(\x_\tau^\wedge)\hat{\mathbf{T}}_k \\
\w(\tau) &= \J(\x_\tau) \dot{\x}_\tau  \\
 &\approx \dot{\x}_\tau  \\
 \dw(\tau) &\approx \J(\x_{\tau}) (\ddot{\x}_{\tau} + \frac{1}{2} \dot{\x}_{\tau}^\curlywedge \w(\tau) ) \\
 &\approx \ddot{\x}_{\tau} 
\end{align*}
$$

where we have made the approximation that $\x_\tau$ is small, and so $\J(\x_{\tau})$ is close to identity. Hence, our extrapolation formulas are approximate and this approximation only holds so long as $\x_\tau$ is small.

$$
\boldsymbol{\Phi} = \begin{bmatrix} \mathbf{C}_{11} & \mathbf{C}_{12} & \mathbf{C}_{13} \\ \mathbf{0} & \mathbf{C}_{22} & \mathbf{C}_{23} \\ \mathbf{0} & \mathbf{0} & \mathbf{C}_{33}  \end{bmatrix}
$$

$\boldsymbol{\Phi}$ is defined differently for the WNOJ and Singer priors.

Next, we can easily define the Jacobians with respect to the perturbations:

$$
\begin{align*}
\frac{\partial \x_\tau}{\partial \delta \w_k} &= \mathbf{C}_{12} \\
\frac{\partial \x_\tau}{\partial \delta \dw_k} &= \mathbf{C}_{13} \\
\frac{\partial \dot{\x}_\tau}{\partial \delta \w_k} &= \mathbf{C}_{22} \\
\frac{\partial \dot{\x}_\tau}{\partial \delta \dw_k} &= \mathbf{C}_{23} \\
\frac{\partial \ddot{\x}_\tau}{\partial \delta \dw_k} &= \mathbf{C}_{33}
\end{align*}
$$
