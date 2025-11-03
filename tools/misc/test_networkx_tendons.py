#!/usr/bin/env python3
"""
Test NetworkX-based tendon network generation.
"""

import sys
sys.path.append('/Users/hassanshahristani/Documents/IIB/4th_year_project/tools')

def test_networkx_graph_generation():
    """Test basic NetworkX graph-based tendon creation."""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️  NetworkX not installed. Install with: pip install networkx")
        print("   Skipping NetworkX tests.")
        return
    
    import mujoco
    from tools.mujoco_sim_template import ModelBuilder, MuJoCoSimulation
    
    # Create base model
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    builder = ModelBuilder.from_xml_path(xml_path)
    
    # Add sites to boxes
    builder.add_site(name="red_s", body_name="red_box", pos=[0, 0, 0], size=0.02)
    builder.add_site(name="green_s", body_name="green_box", pos=[0, 0, 0], size=0.02)
    builder.add_site(name="blue_s", body_name="blue_box", pos=[0, 0, 0], size=0.02)
    
    # Create simple triangle graph
    G = nx.Graph()
    G.add_edge('red_s', 'green_s', stiffness=200, damping=2.0, springlength=0.3)
    G.add_edge('green_s', 'blue_s', stiffness=150, damping=1.5, springlength=0.4)
    G.add_edge('blue_s', 'red_s', stiffness=180, damping=1.8, springlength=0.35)
    
    # Generate tendons from graph
    builder.add_tendons_from_graph(
        graph=G,
        tendon_name_format="graph_{i}_{j}"
    )
    
    # Compile
    model = builder.compile()
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Verify tendons were created
    expected_tendons = [
        "graph_red_s_green_s",
        "graph_green_s_blue_s",
        "graph_blue_s_red_s"
    ]
    
    for tendon_name in expected_tendons:
        try:
            tendon = sim.reg_tendon(f"t_{tendon_name}", tendon_name)
            print(f"✅ Tendon '{tendon_name}' created successfully")
            print(f"   - Length: {tendon.length:.3f}")
            print(f"   - Stiffness: {tendon.stiffness:.1f}")
            print(f"   - Damping: {tendon.damping:.1f}")
        except Exception as e:
            print(f"❌ Failed to find tendon '{tendon_name}': {e}")
            return False
    
    print("\n✅ All NetworkX graph tendons created successfully!")
    return True


def test_edge_attributes():
    """Test that edge attributes are properly applied to tendons."""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️  NetworkX not installed. Skipping test.")
        return
    
    import mujoco
    from tools.mujoco_sim_template import ModelBuilder, MuJoCoSimulation
    
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    builder = ModelBuilder.from_xml_path(xml_path)
    
    # Add sites
    builder.add_site(name="s1", body_name="red_box", pos=[0, 0, 0.05], size=0.02)
    builder.add_site(name="s2", body_name="green_box", pos=[0, 0, -0.05], size=0.02)
    
    # Create graph with specific properties
    G = nx.Graph()
    G.add_edge('s1', 's2',
               name='custom_tendon',
               stiffness=333.3,
               damping=4.5,
               springlength=0.25)
    
    builder.add_tendons_from_graph(G)
    
    model = builder.compile()
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Verify properties
    tendon = sim.reg_tendon("test", "custom_tendon")
    
    print(f"\n🔍 Testing edge attribute propagation:")
    print(f"   Stiffness: {tendon.stiffness:.1f} (expected 333.3)")
    print(f"   Damping: {tendon.damping:.1f} (expected 4.5)")
    print(f"   Spring length: {tendon.springlength:.2f} (expected 0.25)")
    
    # Tolerance check
    assert abs(tendon.stiffness - 333.3) < 0.1, "Stiffness mismatch"
    assert abs(tendon.damping - 4.5) < 0.1, "Damping mismatch"
    assert abs(tendon.springlength - 0.25) < 0.01, "Springlength mismatch"
    
    print("✅ Edge attributes correctly applied!")
    return True


def test_default_parameters():
    """Test that default parameters work correctly."""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️  NetworkX not installed. Skipping test.")
        return
    
    from tools.mujoco_sim_template import ModelBuilder, MuJoCoSimulation
    
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    builder = ModelBuilder.from_xml_path(xml_path)
    
    builder.add_site(name="site_a", body_name="red_box", pos=[0, 0, 0], size=0.02)
    builder.add_site(name="site_b", body_name="blue_box", pos=[0, 0, 0], size=0.02)
    
    # Graph without edge attributes
    G = nx.Graph()
    G.add_edge('site_a', 'site_b')  # No attributes
    
    # Use defaults
    builder.add_tendons_from_graph(
        graph=G,
        default_stiffness=555.0,
        default_damping=7.7,
        default_springlength=0.44,
        tendon_name_format="default_test"
    )
    
    model = builder.compile()
    sim = MuJoCoSimulation.from_builder(builder)
    
    tendon = sim.reg_tendon("test", "default_test")
    
    print(f"\n🔍 Testing default parameters:")
    print(f"   Stiffness: {tendon.stiffness:.1f} (expected 555.0)")
    print(f"   Damping: {tendon.damping:.1f} (expected 7.7)")
    print(f"   Spring length: {tendon.springlength:.2f} (expected 0.44)")
    
    assert abs(tendon.stiffness - 555.0) < 0.1, "Default stiffness not applied"
    assert abs(tendon.damping - 7.7) < 0.1, "Default damping not applied"
    assert abs(tendon.springlength - 0.44) < 0.01, "Default springlength not applied"
    
    print("✅ Default parameters work correctly!")
    return True


def test_multiple_graph_types():
    """Test various NetworkX graph types."""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️  NetworkX not installed. Skipping test.")
        return
    
    from tools.mujoco_sim_template import ModelBuilder
    
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    
    # Test complete graph
    print("\n🔍 Testing complete graph (K3):")
    builder = ModelBuilder.from_xml_path(xml_path)
    for i in range(3):
        builder.add_site(name=f"k{i}", body_name="red_box", 
                        pos=[i*0.05, 0, 0], size=0.01)
    
    G = nx.complete_graph([f"k{i}" for i in range(3)])
    nx.set_edge_attributes(G, 100, 'stiffness')
    builder.add_tendons_from_graph(G, tendon_name_format="k3_{i}_{j}")
    model = builder.compile()
    print(f"   Created {model.ntendon} tendons (expected 3)")
    assert model.ntendon >= 3, "Should have at least 3 tendons"
    
    # Test cycle graph
    print("\n🔍 Testing cycle graph (C4):")
    builder = ModelBuilder.from_xml_path(xml_path)
    for i in range(4):
        builder.add_site(name=f"c{i}", body_name="green_box",
                        pos=[0, i*0.05, 0], size=0.01)
    
    G = nx.cycle_graph([f"c{i}" for i in range(4)])
    nx.set_edge_attributes(G, 150, 'stiffness')
    builder.add_tendons_from_graph(G, tendon_name_format="c4_{i}_{j}")
    model = builder.compile()
    print(f"   Created {model.ntendon} tendons (expected 4)")
    assert model.ntendon >= 4, "Should have at least 4 tendons"
    
    # Test star graph
    print("\n🔍 Testing star graph:")
    builder = ModelBuilder.from_xml_path(xml_path)
    nodes = ['center'] + [f"spoke{i}" for i in range(4)]
    for node in nodes:
        builder.add_site(name=node, body_name="blue_box",
                        pos=[0, 0, 0], size=0.01)
    
    G = nx.star_graph(nodes)
    nx.set_edge_attributes(G, 200, 'stiffness')
    builder.add_tendons_from_graph(G, tendon_name_format="star_{i}_{j}")
    model = builder.compile()
    print(f"   Created {model.ntendon} tendons (expected 4)")
    assert model.ntendon >= 4, "Should have at least 4 tendons"
    
    print("\n✅ All graph types work correctly!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NetworkX-Based Tendon Generation")
    print("=" * 60)
    
    tests = [
        ("Basic Graph Generation", test_networkx_graph_generation),
        ("Edge Attributes", test_edge_attributes),
        ("Default Parameters", test_default_parameters),
        ("Multiple Graph Types", test_multiple_graph_types),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. See details above.")
