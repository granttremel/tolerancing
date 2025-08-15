

from abc import ABC, abstractmethod
import numpy as np
from math import factorial

class OrthonormalBasis(ABC):
    """Base class for all geometry-specific bases"""
    
    dim=-1
    
    @abstractmethod
    def normalize(self, mode_index: int) -> float:
        """Return normalization constant for mode to have unit variance"""
        pass
    
    @abstractmethod
    def evaluate(self, coords: np.ndarray, mode: tuple) -> tuple:
        """
        Evaluate basis function and its derivatives
        Returns: (value, d/du, d/dv) for surface gradients
        """
        pass
    
    def sample_surface(self, coords: np.ndarray, 
                                   frame: np.ndarray,
                                   mode_weights: dict, 
                                   rng) -> tuple:
        """
        Sample surface perturbation including normal rotation
        
        Args:
            coords: Parametric coordinates
            frame: [tangent1, tangent2, normal] or [tangent, normal1, normal2]
            mode_weights: Modal amplitudes
            
        Returns:
            (perturbed_points, perturbed_frame)
        """
        n_points = len(coords)
        dim = frame.shape[0]  # 2 for line, 3 for surface
        
        # Sample the displacement field
        if dim == 2:  # Line: needs 2D deviation
            # Two independent deviation fields (v and w directions)
            dev_v = self._sample_1d_field(coords, mode_weights['v'], rng)
            dev_w = self._sample_1d_field(coords, mode_weights['w'], rng)
            
            # Compute derivatives for normal rotation
            ddev_v_du = self._sample_1d_derivatives(coords, mode_weights['v'], rng)
            ddev_w_du = self._sample_1d_derivatives(coords, mode_weights['w'], rng)
            
            # Perturb normals based on derivatives
            perturbed_frames = []
            for i in range(n_points):
                # The tangent tilts based on transverse displacement gradients
                tangent = frame[0]  # Original tangent
                normal_v = frame[1] 
                normal_w = frame[2]
                
                # Small angle approximation for rotation
                new_tangent = tangent + ddev_v_du[i] * normal_v + ddev_w_du[i] * normal_w
                new_tangent /= np.linalg.norm(new_tangent)
                
                # Recompute orthogonal frame
                new_normal_v = normal_v - np.dot(normal_v, new_tangent) * new_tangent
                new_normal_v /= np.linalg.norm(new_normal_v)
                new_normal_w = np.cross(new_tangent, new_normal_v)
                
                perturbed_frames.append([new_tangent, new_normal_v, new_normal_w])
                
            displacements = dev_v[:, np.newaxis] * frame[1] + dev_w[:, np.newaxis] * frame[2]
            
        else:  # Surface: single scalar field
            dev = self._sample_2d_field(coords, mode_weights, rng)
            ddev_du, ddev_dv = self._sample_2d_derivatives(coords, mode_weights, rng)
            
            perturbed_frames = []
            for i in range(n_points):
                tangent_u = frame[0]
                tangent_v = frame[1]
                normal = frame[2]
                
                # Surface gradient tilts the normal
                # ∇z = (∂z/∂u, ∂z/∂v) in tangent plane
                gradient = ddev_du[i] * tangent_u + ddev_dv[i] * tangent_v
                
                # New normal (small angle approximation)
                new_normal = normal - gradient
                new_normal /= np.linalg.norm(new_normal)
                
                # Recompute tangent frame
                new_tangent_u = tangent_u - np.dot(tangent_u, new_normal) * new_normal
                new_tangent_u /= np.linalg.norm(new_tangent_u)
                new_tangent_v = np.cross(new_normal, new_tangent_u)
                
                perturbed_frames.append([new_tangent_u, new_tangent_v, new_normal])
                
            displacements = dev[:, np.newaxis] * frame[2]  # Along normal
            
        return displacements, perturbed_frames
class NullBasis(OrthonormalBasis):
    dim=0
    
    def normalize(self, index:int) -> float:
        return 1.0
    
    def evaluate(self, coords: np.ndarray, mode: tuple) -> np.ndarray:
        return 0.0

class ZernikeBasis(OrthonormalBasis):
    """Zernike polynomials for circular domains (planes)"""
    
    def __init__(self):
        # Cache frequently used modes
        self._cache = {}
        
    def normalize(self, n: int, m: int = 0) -> float:
        """
        Zernike R_n^m normalization
        ∫∫ [R_n^m(ρ)]^2 ρ dρ dθ = π/(2n+2) for m≠0, 2π/(2n+2) for m=0
        """
        if m == 0:
            return np.sqrt(2 * (n + 1) / np.pi)
        else:
            return np.sqrt(4 * (n + 1) / np.pi)
    
    def _radial_polynomial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Compute Zernike radial polynomial R_n^|m|"""
        m = abs(m)
        if (n - m) % 2 != 0:
            return np.zeros_like(rho)
        
        result = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            coeff = ((-1)**k * factorial(n - k) / 
                    (factorial(k) * factorial((n + m)//2 - k) * 
                     factorial((n - m)//2 - k)))
            result += coeff * rho**(n - 2*k)
        
        return result
    
    def evaluate(self, coords: np.ndarray, mode: tuple) -> np.ndarray:
        """
        Evaluate Zernike mode (n, m) at coordinates
        coords: [[ρ, θ], ...] in polar coordinates
        mode: (n, m) indices
        """
        n, m = mode
        rho = coords[:, 0]
        theta = coords[:, 1]
        
        R_nm = self._radial_polynomial(n, abs(m), rho)
        
        if m >= 0:
            return R_nm * np.cos(m * theta)
        else:
            return R_nm * np.sin(abs(m) * theta)
    
    def get_standard_modes(self) -> dict:
        """Return standard Zernike modes with optical names"""
        return {
            (0, 0): 'piston',
            (1, -1): 'tilt_y',
            (1, 1): 'tilt_x',
            (2, -2): 'astigmatism_45',
            (2, 0): 'defocus',
            (2, 2): 'astigmatism_0',
            (3, -3): 'trefoil_30',
            (3, -1): 'coma_y',
            (3, 1): 'coma_x',
            (3, 3): 'trefoil_0',
            (4, 0): 'spherical',
            # Higher orders as needed
        }
    
class ChebyshevBasis(OrthonormalBasis):
    """For line/axis - normalized Chebyshev on [-1,1]"""

    def normalize(self, n: int) -> float:
        # ∫_{-1}^{1} T_n^2(x) dx = π/2 for n>0, π for n=0
        if n == 0:
            return 1.0 / np.sqrt(np.pi)
        else:
            return np.sqrt(2.0 / np.pi)

    def evaluate(self, u: np.ndarray, n: int) -> np.ndarray:
        # u should be normalized to [-1, 1]
        return np.cos(n * np.arccos(u))


class FourierBasis(OrthonormalBasis):
    """For periodic dimension (θ in cylinder)"""
    
    def normalize(self, m: int) -> float:
        # ∫_0^{2π} cos^2(mθ) dθ = π
        return 1.0 / np.sqrt(np.pi)
    
    def evaluate(self, theta: np.ndarray, m: int) -> np.ndarray:
        return np.cos(m * theta)


class CylindricalBasis(OrthonormalBasis):
    """For periodic dimension (θ in cylinder)"""
    
    def __init__(self):
        super().__init__()
        self._cheby = ChebyshevBasis()
        self._four = FourierBasis()
    
    def normalize(self, n:int, m: int) -> float:
        
        self._cheby.normalize(n)
        self._four.normalize(m)
        
        # ∫_0^{2π} cos^2(mθ) dθ = π
        return 1.0 / np.sqrt(np.pi)
    
    def evaluate(self, theta: np.ndarray, m: int) -> np.ndarray:
        return np.cos(m * theta)
    

class SphericalHarmonicBasis(OrthonormalBasis):
    """For sphere - already orthonormal!"""
    
    def normalize(self, l: int, m: int = 0) -> float:
        # Spherical harmonics are already normalized
        # ∫∫ |Y_l^m|^2 sin(θ) dθ dφ = 1
        return 1.0
    
    def evaluate(self, coords: np.ndarray, l: int, m: int = 0) -> np.ndarray:
        theta, phi = coords[:, 0], coords[:, 1]
        from scipy.special import sph_harm
        return sph_harm(m, l, phi, theta).real