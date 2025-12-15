use crate::types::{SystemId, AccessSets};
use crate::manager::ECSManager;

pub trait System: Send + Sync {
    fn id(&self) -> SystemID;
    fn name(&self) -> &'static str;
    fn access(&self) -> AccessSets;
    fn run(&self, ecs_manager: &mut ECSManager);
}


pub struct FnSystem<F>
where F: Fn(&mut ECSManager) + Send + Sync + 'static
{
    id: SystemID,
    name: &'static str,
    access: AccessSets,
    f: F,
}

impl<F> FnSystem<F>
where F: Fn(&mut ECSManager) + Send + Sync + 'static
{
    pub fn new(id: SystemID, name: &'static str, access: AccessSets, f: F) -> Self {
        Self { id, name, access, f }
    }
}

impl<F> System for FnSystem<F>
where F: Fn(&mut ECSManager) + Send + Sync + 'static
{
    fn id(&self) -> SystemID { self.id }
    fn name(&self) -> &'static str { self.name }
    fn access(&self) -> AccessSets { self.access.clone() }
    fn run(&self, ecs_manager: &mut ECSManager) { (self.f)(ecs_manager) }
}
