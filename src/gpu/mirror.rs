#![cfg(feature = "gpu")]

use std::collections::HashMap;

use crate::engine::archetype::Archetype;
use crate::engine::component::{iter_bits_from_words, ComponentRegistry, Signature};
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::types::{ArchetypeID, ChunkID, ComponentID};

#[cfg(feature = "gpu")]
use crate::engine::dirty::DirtyChunks;

use crate::gpu::GPUContext;

#[inline]
fn align_to_4(bytes: usize) -> usize {
    (bytes + 3) & !3
}

#[derive(Debug)]
struct BufferEntry {
    buffer: wgpu::Buffer,
    bytes: usize,
}

#[derive(Debug)]
struct ReadbackEntry {
    buffer: wgpu::Buffer,
    bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct DownloadTask {
    archetype_index: usize,
    archetype_id: ArchetypeID,
    component_id: ComponentID,
    bytes_total: usize,
}

#[derive(Debug)]
pub struct Mirror {
    buffers: HashMap<(ArchetypeID, ComponentID), BufferEntry>,
    readback_buffers: HashMap<(ArchetypeID, ComponentID), ReadbackEntry>,
    binding_generation: u64,
}

impl Mirror {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            readback_buffers: HashMap::new(),
            binding_generation: 0,
        }
    }

    /// Monotonic generation for component buffers used in GPU bind groups.
    #[inline]
    pub(crate) fn binding_generation(&self) -> u64 {
        self.binding_generation
    }

    /// Checks that the component identified by `component_id` is registered and GPU-safe,
    /// returning its element size in bytes and its name. Uses the provided `ComponentRegistry`
    /// to look up the component description.
    ///
    /// # Errors
    /// Returns an error if the component is not registered or if it is not marked for GPU usage.
    pub fn ensure_gpu_safe(
        &self,
        component_id: ComponentID,
        registry: &ComponentRegistry,
    ) -> ECSResult<(usize, &'static str)> {
        let description = registry
            .description_by_component_id(component_id)
            .ok_or_else(|| ECSError::from(ExecutionError::MissingComponent { component_id }))?;

        if !description.gpu_usage {
            return Err(ECSError::from(ExecutionError::GpuUnsupportedComponent {
                component_id,
                name: description.name,
            }));
        }
        Ok((description.size, description.name))
    }

    /// Upload dirty chunks for the required signature.
    pub fn upload_signature_dirty_chunks(
        &mut self,
        context: &GPUContext,
        archetypes: &[Archetype],
        signature: &Signature,
        dirty: &DirtyChunks,
        registry: &ComponentRegistry,
    ) -> ECSResult<()> {
        let mut component_ids: Vec<ComponentID> =
            iter_bits_from_words(&signature.components).collect();
        component_ids.sort_unstable();

        for archetype in archetypes {
            let chunk_count = archetype.chunk_count()?;
            if chunk_count == 0 {
                continue;
            }

            for &component_id in &component_ids {
                if archetype.has(component_id) {
                    let (size, _) = self.ensure_gpu_safe(component_id, registry)?;
                    self.upload_column_dirty_chunks(context, archetype, component_id, size, dirty)?;
                }
            }
        }

        Ok(())
    }

    pub fn download_signature(
        &mut self,
        context: &GPUContext,
        archetypes: &mut [Archetype],
        signature: &Signature,
        registry: &ComponentRegistry,
    ) -> ECSResult<()> {
        let mut component_ids: Vec<ComponentID> =
            iter_bits_from_words(&signature.components).collect();
        component_ids.sort_unstable();

        let mut tasks = Vec::new();

        for (archetype_index, archetype) in archetypes.iter().enumerate() {
            for &component_id in &component_ids {
                if archetype.has(component_id) {
                    let (size, _) = self.ensure_gpu_safe(component_id, registry)?;
                    let len = archetype.length()?;
                    if len == 0 {
                        continue;
                    }

                    let bytes_total = align_to_4(len * size);
                    self.ensure_buffer(
                        context,
                        archetype.archetype_id(),
                        component_id,
                        bytes_total,
                    );
                    self.ensure_readback_buffer(
                        context,
                        archetype.archetype_id(),
                        component_id,
                        bytes_total,
                    );
                    tasks.push(DownloadTask {
                        archetype_index,
                        archetype_id: archetype.archetype_id(),
                        component_id,
                        bytes_total,
                    });
                }
            }
        }

        self.download_columns_batched(context, archetypes, &tasks)?;
        Ok(())
    }

    pub fn buffer_for(
        &self,
        archetype: ArchetypeID,
        component_id: ComponentID,
    ) -> Option<&wgpu::Buffer> {
        self.buffers
            .get(&(archetype, component_id))
            .map(|e| &e.buffer)
    }

    fn ensure_buffer(
        &mut self,
        context: &GPUContext,
        archetype: ArchetypeID,
        component_id: ComponentID,
        bytes: usize,
    ) {
        let key = (archetype, component_id);

        let recreate = match self.buffers.get(&key) {
            None => true,
            Some(e) => e.bytes < bytes,
        };

        if recreate {
            let label = format!("abm_component_storage[a{} c{}]", archetype, component_id);

            let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.buffers.insert(key, BufferEntry { buffer, bytes });
            self.binding_generation = self.binding_generation.wrapping_add(1);
        }
    }

    fn ensure_readback_buffer(
        &mut self,
        context: &GPUContext,
        archetype: ArchetypeID,
        component_id: ComponentID,
        bytes: usize,
    ) {
        let key = (archetype, component_id);

        let recreate = match self.readback_buffers.get(&key) {
            None => true,
            Some(e) => e.bytes < bytes,
        };

        if recreate {
            let label = format!("abm_readback_staging[a{} c{}]", archetype, component_id);

            let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.readback_buffers
                .insert(key, ReadbackEntry { buffer, bytes });
        }
    }

    fn upload_column_dirty_chunks(
        &mut self,
        context: &GPUContext,
        archetype: &Archetype,
        component_id: ComponentID,
        component_size: usize,
        dirty: &DirtyChunks,
    ) -> ECSResult<()> {
        let len = archetype.length()?;
        if len == 0 {
            return Ok(());
        }

        let chunk_count = archetype.chunk_count()?;
        if chunk_count == 0 {
            return Ok(());
        }
        let require_aligned = (component_size % 4) == 0;

        let bytes_total = align_to_4(len * component_size);
        self.ensure_buffer(context, archetype.archetype_id(), component_id, bytes_total);

        // Determine dirty chunks and clear them in the tracker.
        let mut dirty_chunks =
            dirty.take_dirty_chunks(archetype.archetype_id(), component_id, chunk_count);

        if dirty_chunks.is_empty() {
            return Ok(());
        }

        // Sort for deterministic upload order.
        dirty_chunks.sort_unstable();

        if !require_aligned {
            return self.upload_column_full(context, archetype, component_id, component_size);
        }

        let storage = self
            .buffers
            .get(&(archetype.archetype_id(), component_id))
            .unwrap();

        let locked = archetype
            .component_locked(component_id)
            .ok_or_else(|| ECSError::from(ExecutionError::MissingComponent { component_id }))?;
        let guard = locked.read().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "attribute read lock (upload)",
            })
        })?;

        let mut prefix_rows: Vec<usize> = Vec::with_capacity(chunk_count + 1);
        prefix_rows.push(0);
        for c in 0..chunk_count {
            let v = archetype.chunk_valid_length(c)?;
            let prev = *prefix_rows.last().unwrap();
            prefix_rows.push(prev + v);
        }

        // Upload each dirty chunk slice into the correct offset in the packed GPU buffer.
        for &chunk in &dirty_chunks {
            let valid = archetype.chunk_valid_length(chunk)?;
            if valid == 0 {
                continue;
            }

            let chunk_id: ChunkID = chunk
                .try_into()
                .map_err(|_| ECSError::from(ExecutionError::InternalExecutionError))?;

            let (ptr, bytes) = guard
                .chunk_bytes(chunk_id, valid)
                .ok_or_else(|| ECSError::from(ExecutionError::InternalExecutionError))?;

            // Calculate offset into packed buffer
            let row_off = prefix_rows[chunk];
            let byte_off = row_off * component_size;

            // Ensure alignment for write_buffer
            if (byte_off % 4) != 0 || (bytes % 4) != 0 {
                return self.upload_column_full(context, archetype, component_id, component_size);
            }

            let src = unsafe { std::slice::from_raw_parts(ptr, bytes) };
            context
                .queue
                .write_buffer(&storage.buffer, byte_off as u64, src);
        }

        Ok(())
    }

    fn upload_column_full(
        &mut self,
        context: &GPUContext,
        archetype: &Archetype,
        component_id: ComponentID,
        component_size: usize,
    ) -> ECSResult<()> {
        let len = archetype.length()?;
        if len == 0 {
            return Ok(());
        }

        let bytes_total = align_to_4(len * component_size);
        self.ensure_buffer(context, archetype.archetype_id(), component_id, bytes_total);

        let storage = self
            .buffers
            .get(&(archetype.archetype_id(), component_id))
            .unwrap();

        let locked = archetype
            .component_locked(component_id)
            .ok_or_else(|| ECSError::from(ExecutionError::MissingComponent { component_id }))?;
        let guard = locked.read().map_err(|_| {
            ECSError::from(ExecutionError::LockPoisoned {
                what: "attribute read lock (full upload)",
            })
        })?;

        let mut host = vec![0u8; bytes_total];
        let mut offset = 0usize;

        let chunks = archetype.chunk_count()?;
        for chunk in 0..chunks {
            let valid = archetype.chunk_valid_length(chunk)?;
            if valid == 0 {
                continue;
            }

            let (pointer, bytes) = guard
                .chunk_bytes(chunk as ChunkID, valid)
                .ok_or_else(|| ECSError::from(ExecutionError::InternalExecutionError))?;
            let source = unsafe { std::slice::from_raw_parts(pointer, bytes) };

            host[offset..offset + bytes].copy_from_slice(source);
            offset += bytes;
        }

        context.queue.write_buffer(&storage.buffer, 0, &host);
        Ok(())
    }

    fn download_columns_batched(
        &mut self,
        context: &GPUContext,
        archetypes: &mut [Archetype],
        tasks: &[DownloadTask],
    ) -> ECSResult<()> {
        if tasks.is_empty() {
            return Ok(());
        }

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("abm_readback_encoder"),
            });

        for task in tasks {
            let storage = self
                .buffers
                .get(&(task.archetype_id, task.component_id))
                .unwrap();
            let staging = self
                .readback_buffers
                .get(&(task.archetype_id, task.component_id))
                .unwrap();
            encoder.copy_buffer_to_buffer(
                &storage.buffer,
                0,
                &staging.buffer,
                0,
                task.bytes_total as u64,
            );
        }

        context.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::with_capacity(tasks.len());
        for task in tasks {
            let staging = self
                .readback_buffers
                .get(&(task.archetype_id, task.component_id))
                .unwrap();
            let slice = staging.buffer.slice(0..task.bytes_total as u64);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            receivers.push(receiver);
        }

        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("wgpu poll failed during map_async: {e:?}").into(),
                })
            })?;

        for (task, receiver) in tasks.iter().zip(receivers) {
            receiver.recv().ok().transpose().map_err(|_| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: "failed to map readback buffer".into(),
                })
            })?;

            let staging = self
                .readback_buffers
                .get(&(task.archetype_id, task.component_id))
                .unwrap();
            let slice = staging.buffer.slice(0..task.bytes_total as u64);
            let data = slice.get_mapped_range();
            let host: &[u8] = &data;

            let archetype = &mut archetypes[task.archetype_index];
            let locked = archetype
                .component_locked(task.component_id)
                .ok_or_else(|| {
                    ECSError::from(ExecutionError::MissingComponent {
                        component_id: task.component_id,
                    })
                })?;
            let mut guard = locked.write().map_err(|_| {
                ECSError::from(ExecutionError::LockPoisoned {
                    what: "attribute write lock (download)",
                })
            })?;

            let mut offset = 0usize;
            let chunks = archetype.chunk_count()?;
            for chunk in 0..chunks {
                let valid = archetype.chunk_valid_length(chunk)?;
                if valid == 0 {
                    continue;
                }

                let (pointer, bytes) = guard
                    .chunk_bytes_mut(chunk as ChunkID, valid)
                    .ok_or_else(|| ECSError::from(ExecutionError::InternalExecutionError))?;
                let destination = unsafe { std::slice::from_raw_parts_mut(pointer, bytes) };

                destination.copy_from_slice(&host[offset..offset + bytes]);
                offset += bytes;
            }

            drop(data);
            staging.buffer.unmap();
        }
        Ok(())
    }
}
