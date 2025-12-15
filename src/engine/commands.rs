use std::any::Any;

use crate::entity::Entity;
use crate::types::ComponentID;


pub enum Command {
    Spawn { shard: ShardID, archetype: ArchetypeID },
    Despawn { entity: Entity },
    Add { entity: Entity, cid: ComponentID, value: Box<dyn Any + Send> },
    Remove { entity: Entity, cid: ComponentID },
}
