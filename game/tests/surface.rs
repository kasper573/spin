use game::core::surface::SurfaceGrid;

#[test]
fn a_blob_of_points_yields_a_closed_shell() {
    let mut grid = SurfaceGrid::new([-1.0; 3], [1.0; 3], 0.08);
    grid.clear();
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                let p = [
                    i as f32 * 0.1 - 0.25,
                    j as f32 * 0.1 - 0.25,
                    k as f32 * 0.1 - 0.25,
                ];
                grid.splat(p, 0.5, 0.16);
            }
        }
    }
    let mesh = grid.extract(0.9);
    assert!(!mesh.positions.is_empty());
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    assert_eq!(mesh.positions.len(), mesh.foam.len());
    assert_eq!(mesh.indices.len() % 3, 0);
    assert!(
        mesh.indices
            .iter()
            .all(|i| (*i as usize) < mesh.positions.len())
    );
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        assert!(p.iter().all(|v| v.abs() < 0.6), "{p:?}");
        let outward = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        assert!(outward > 0.0, "normal {n:?} at {p:?} points inward");
        assert!((mesh.foam[0] - 0.5).abs() < 1e-4);
    }
}

#[test]
fn an_empty_grid_yields_nothing() {
    let mut grid = SurfaceGrid::new([-1.0; 3], [1.0; 3], 0.1);
    grid.clear();
    let mesh = grid.extract(0.5);
    assert!(mesh.positions.is_empty());
    assert!(mesh.indices.is_empty());
}
