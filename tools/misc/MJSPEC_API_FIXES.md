# MjSpec API Fixes - Changelog

## Summary
Fixed ModelBuilder to work with MuJoCo's MjSpec API. The initial implementation assumed API methods that don't exist; this update uses the correct MjSpec interface.

## API Corrections

### 1. Finding Bodies and Sites
**Before (WRONG):**
```python
body = self.spec.find_body(body_name)  # ❌ Method doesn't exist
site = self.spec.find_site(site_name)  # ❌ Method doesn't exist
```

**After (CORRECT):**
```python
# Implemented recursive search functions
def _find_body(self, name: str, body=None):
    """Recursively search body tree"""
    if body is None:
        body = self.spec.worldbody
    if hasattr(body, 'name') and body.name == name:
        return body
    if hasattr(body, 'bodies'):
        for child in body.bodies:
            result = self._find_body(name, child)
            if result:
                return result
    return None
```

### 2. Site Size Property
**Before (WRONG):**
```python
site.size = [size]  # ❌ Expects 3 elements
```

**After (CORRECT):**
```python
site.size = [size, size, size]  # ✓ MjsSite.size needs [x, y, z]
```

### 3. Tendon Creation
**Before (WRONG):**
```python
tendon.kind = mujoco.mjtTrn.mjTRN_JOINT  # ❌ Attribute doesn't exist
wrap = tendon.add_wrap()                  # ❌ Method doesn't exist
wrap.site = site_spec                     # ❌ Wrong API
```

**After (CORRECT):**
```python
# No need to set kind - spatial tendons are defined by wrapping sites
tendon.wrap_site(site_name)  # ✓ Takes string name, not object
```

### 4. Tendon Properties
**Before (WRONG):**
```python
tendon.stiffness = [stiffness]  # ❌ Takes float, not list
```

**After (CORRECT):**
```python
tendon.stiffness = stiffness  # ✓ Direct float assignment
tendon.damping = damping
tendon.springlength = springlength
```

### 5. Body Inertial Properties
**Before (WRONG):**
```python
inertial = body.add_inertial()  # ❌ Method doesn't exist
inertial.mass = mass
```

**After (CORRECT):**
```python
body.mass = mass  # ✓ Direct attribute assignment
```

### 6. Free Joint Creation
**Before (WRONG):**
```python
joint = body.add_joint()
joint.name = f"{name}__free_joint"
joint.type = mujoco.mjtJoint.mjJNT_FREE
```

**After (CORRECT):**
```python
body.add_freejoint()  # ✓ Dedicated method for free joints
```

### 7. Geom Size Property
**Before (WRONG):**
```python
geom.size = [0.08]  # ❌ For sphere - expects 3 elements
```

**After (CORRECT):**
```python
# Pad to 3 elements
if len(geom_size) == 1:
    geom.size = [geom_size[0], geom_size[0], geom_size[0]]
elif len(geom_size) == 2:
    geom.size = [geom_size[0], geom_size[1], 0]
else:
    geom.size = geom_size
```

### 8. Tendon Force Property
**Before (WRONG):**
```python
@property
def force(self) -> float:
    return self.data.ten_force[self.id]  # ❌ ten_force doesn't exist!
```

**After (CORRECT):**
```python
@property
def force(self) -> float:
    """Computed from spring/damper properties"""
    sl = self.springlength[0] if isinstance(self.springlength, np.ndarray) else self.springlength
    k = self.stiffness[0] if isinstance(self.stiffness, np.ndarray) else self.stiffness
    d = self.damping[0] if isinstance(self.damping, np.ndarray) else self.damping
    
    stretch = self.length - sl if sl > 0 else 0
    return k * stretch + d * self.velocity
```

## MjSpec API Reference

### Correct Methods Available

**MjSpec:**
- `.worldbody` - Root body
- `.add_tendon()` - Create tendon

**MjsBody:**
- `.add_body()` - Add child body
- `.add_geom()` - Add geometry
- `.add_site()` - Add site
- `.add_freejoint()` - Add free joint (6 DOF)
- `.mass` - Direct attribute
- `.pos`, `.quat` - Direct attributes

**MjsSite:**
- `.name`, `.pos`, `.quat`, `.rgba` - Direct attributes
- `.size` - Requires [x, y, z] array

**MjsGeom:**
- `.name`, `.type`, `.size`, `.rgba` - Direct attributes
- `.size` - Requires [x, y, z] array

**MjsTendon:**
- `.name`, `.stiffness`, `.damping`, `.springlength`, `.rgba` - Direct attributes
- `.wrap_site(site_name: str)` - Add site to tendon path (takes string name!)
- `.wrap_geom(geom_name: str)` - Add geom to tendon path
- `.wrap_joint(joint_name: str)` - Wrap around joint
- `.wrap_pulley(divisor: float)` - Add pulley

### Data Arrays Available

**MjData tendon arrays:**
- `.ten_length` - Tendon lengths
- `.ten_velocity` - Tendon velocities
- ❌ `.ten_force` - DOES NOT EXIST (must compute from spring law)

**MjModel tendon arrays:**
- `.tendon_stiffness` - Spring stiffness (may be array)
- `.tendon_damping` - Damping coefficient (may be array)
- `.tendon_lengthspring` - Rest length (may be array)

## Testing

All tests now pass:
```bash
$ python tools/test_model_builder.py
Testing ModelBuilder...
  ✓ Adding sites...
  ✓ Adding tendon...
  ✓ Adding new body...
  ✓ Compiling...
  ✓ Registering entities...
  ✓ Checking properties...
  ✓ Running simulation steps...
  ✓ Tendon length: 0.300m
  ✓ Tendon force: 0.00N
  ✓ Body positions: A=0.498, B=0.498, C=0.498

✅ All tests passed!
```

## Key Takeaways

1. **MjSpec uses direct attribute assignment** for most properties, not setter methods
2. **String names, not objects** - `wrap_site` takes the site name as a string
3. **Arrays need correct dimensions** - Sites and geoms need 3-element size arrays
4. **No `ten_force` in MjData** - Must compute from spring/damper law
5. **Recursive search needed** - No built-in find methods for bodies/sites
6. **Properties may be arrays** - Tendon stiffness/damping/springlength can be multi-dimensional

## Files Modified

- `tools/mujoco_sim_template.py` - Fixed all ModelBuilder and Tendon API calls
- `tools/test_model_builder.py` - Verification tests (all passing)

## Usage Now Works

```python
from tools.mujoco_sim_template import ModelBuilder, MuJoCoSimulation

# Load and modify model
builder = ModelBuilder()
builder.from_xml_path("robot.xml")

# Add sites (with correct 3-element size)
builder.add_site("hand_tip", "hand_link", pos=[0, 0, 0.05])

# Add tendon (with string site names)
builder.add_tendon("spring", sites=["hand_tip", "target_site"],
                  stiffness=100.0, damping=5.0)

# Add body (with correct API)
builder.add_body("box", pos=[0, 0, 0.5], geom_type="sphere", geom_size=[0.08])

# Compile and simulate
sim = MuJoCoSimulation.from_builder(builder)
tendon = sim.reg_tendon("spring", "spring")

print(f"Force: {tendon.force}N")  # Computed from spring law
```
