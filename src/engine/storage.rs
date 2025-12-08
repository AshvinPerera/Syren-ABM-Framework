use std::{
    ptr,
    any::{Any, TypeId},
    any::{type_name, type_name_of_val},
    mem::MaybeUninit,
    convert::TryInto
};

use crate::types::{ChunkID, RowID, CHUNK_CAP};
use crate::error::{PositionOutOfBoundsError, TypeMismatchError, AttributeError};


pub trait TypeErasedAttribute: Any + Send + Sync {
    fn chunk_count(&self) -> usize;
    fn length(&self) -> usize;
    fn last_chunk_length(&self) -> usize;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any; 

    fn chunk_slice_ref<T: 'static>(&self, chunk: ChunkID, length: usize) -> Option<&[T]>;
    fn chunk_slice_mut<T: 'static>(&mut self, chunk: ChunkID, length: usize) -> Option<&mut [T]>;

    fn element_type_id(&self) -> TypeId;
    fn element_type_name(&self) -> &'static str;

    fn swap_remove(&mut self, chunk: ChunkID, row: RowID) -> Result<Option<(ChunkID, RowID)>, AttributeError>;
    fn push_dyn(&mut self, value: Box<dyn Any>) -> Result<(ChunkID, RowID), AttributeError>;
    fn push_from(&mut self, source: &mut dyn TypeErasedAttribute, source_chunk: ChunkID, source_row: RowID) -> Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), AttributeError>;
}

/// Invariant:
/// - All chunks before the last are fully initialized (CHUNK_CAP elements).
/// - Only the last chunk may be partially initialized with `last_chunk_length`.
/// - `length` is the total number of initialized elements.
pub struct Attribute<T> {
    chunks: Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>,
    last_chunk_length: usize, // number of initialized elements in the last chunk
    length: usize
}

impl<T> Default for Attribute<T> {
    fn default() -> Self {
        Self { chunks: Vec::new(), last_chunk_length: 0, length: 0 }
    }
}

impl<T> Attribute<T> {
    #[inline]
    fn ensure_last_chunk(&mut self) {
        if self.chunks.is_empty() || self.last_chunk_length == CHUNK_CAP {
            self.chunks.push(Box::new(std::array::from_fn(|_| MaybeUninit::<T>::uninit())));
            self.last_chunk_length = 0;
        }
    }

    #[inline]
    fn get_chunk_position(&self, index: usize) -> (ChunkID, RowID) {
        let chunk = (index / CHUNK_CAP) as ChunkID;
        let row = (index % CHUNK_CAP) as RowID;
        (chunk, row)
    }

    #[inline]
    fn get_slot_unchecked(&mut self, chunk: usize, row: usize) -> &mut MaybeUninit<T> {
        debug_assert!(chunk < self.chunk_count());
        debug_assert!(row < CHUNK_CAP);
        &mut self.chunks[chunk][row]
    }

    #[inline]
    fn to_ids(chunk: usize, row: usize) -> Result<(ChunkID, RowID), AttributeError> {
        let chunk: ChunkID = chunk.try_into().map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row: RowID     = row.try_into().map_err(|_| AttributeError::IndexOverflow("RowID"))?;
        Ok((chunk, row))
    }  

    #[inline]
    fn valid_position(&self, chunk: ChunkID, row: RowID) -> bool {
        let chunk = chunk as usize;
        let row = row as usize;
        if chunk >= self.chunk_count() { return false; }
        if chunk + 1 == self.chunk_count() {
            row < self.last_chunk_length
        } else {
            row < CHUNK_CAP
        }
    }

    #[inline]
    fn last_filled_position(&self) -> Option<(usize, usize)> {
        if self.length == 0 { return None; }
        let index = self.length - 1;
        Some((index / CHUNK_CAP, index % CHUNK_CAP))
    }

    pub fn reserve_chunks(&mut self, additional: usize) {
        self.chunks.reserve(additional);
    }  

    pub fn get(&self, chunk: ChunkID, row: RowID) -> Option<&T> {
        if !self.valid_position(chunk, row) { return None; }
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_ref() })
    }

    pub fn get_mut(&mut self, chunk: ChunkID, row: RowID) -> Option<&mut T> {
        if !self.valid_position(chunk, row) { return None; }
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_mut() })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().enumerate().flat_map(move |(i, chunk)| {
            let initialized = if i == self.chunks.len() - 1 {
                self.last_chunk_length
            } else {
                CHUNK_CAP
            };
            chunk[..initialized]
                .iter()
                .map(|mu| unsafe { mu.assume_init_ref() })
        })
    }

    pub fn push(&mut self, value: T) -> Result<(ChunkID, RowID), AttributeError> {
        self.ensure_last_chunk();
        let chunk_index = self.chunks.len() - 1;
        let row_index   = self.last_chunk_length;

        if chunk_index > ChunkID::MAX as usize || row_index > RowID::MAX as usize {
            if row_index == 0 {
                self.chunks.pop();
                if let Some(last_chunk) = self.chunks.last() {
                    self.last_chunk_length = CHUNK_CAP;
                }
            }
            return Err(AttributeError::IndexOverflow("ChunkID"));
        }

        unsafe {
            self.get_slot_unchecked(chunk_index, row_index)
                .as_mut_ptr()
                .write(value);
        }
        self.last_chunk_length += 1;
        self.length += 1;

        let (chunk_id, row_id) = Self::to_ids(chunk_index, row_index)?;
        Ok((chunk_id, row_id))
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) -> Result<(), AttributeError> {
        for v in iterator {
            self.push(v)?;
        }
        Ok(())
    }

    fn drop_all_initialized_elements(&mut self) {
        if self.length == 0 { return; }

        let mut remaining = self.length;
        for (chunk_idx, chunk) in self.chunks.iter_mut().enumerate() {
            let init_in_chunk = if chunk_idx == self.chunks.len() - 1 {
                self.last_chunk_length
            } else {
                CHUNK_CAP
            };

            let to_drop = init_in_chunk.min(remaining);
            for i in 0..to_drop {
                unsafe { chunk[i].assume_init_drop(); }
            }
            if remaining <= init_in_chunk { break; }
            remaining -= init_in_chunk;
        }
    }

    pub fn clear(&mut self) {
        if self.length == 0 { return; }

        self.drop_all_initialized_elements();
        self.chunks.clear();
        self.length = 0;
        self.last_chunk_length = 0;
    }    

}

impl<T: 'static + Send + Sync> TypeErasedAttribute for Attribute<T> {
    fn chunk_count(&self) -> usize { self.chunks.len() }
    fn length(&self) -> usize { self.length }
    fn last_chunk_length(&self) -> usize { self.last_chunk_length }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn chunk_slice_ref<U: 'static>(&self, chunk: ChunkID, valid_length: usize) -> Option<&[U]> {
        if TypeId::of::<T>() != TypeId::of::<U>() { return None; }
        let slice = &self.chunks[chunk as usize][..];

        let length = if (chunk as usize) == self.chunks.len() - 1 { self.last_chunk_length } else { CHUNK_CAP };
        let length = length.min(valid_length);
        let pointer = slice.as_ptr() as *const T;
        Some(unsafe { std::slice::from_raw_parts(pointer, length) }.as_ref().map(|s| unsafe {
            &*(s as *const [T] as *const [U])
        }).unwrap())
    }

    fn chunk_slice_mut<U: 'static>(&mut self, chunk: ChunkID, valid_length: usize) -> Option<&mut [U]> {
        if TypeId::of::<T>() != TypeId::of::<U>() { return None; }
        let slice = &mut self.chunks[chunk as usize][..];
        let length = if (chunk as usize) == self.chunks.len() - 1 { self.last_chunk_length } else { CHUNK_CAP };
        let length = length.min(valid_length);
        let pointer = slice.as_mut_ptr() as *mut T;
        Some(unsafe { std::slice::from_raw_parts_mut(pointer, length) }.as_mut().map(|s| unsafe {
            &mut *(s as *mut [T] as *mut [U])
        }).unwrap())
    }

    fn element_type_id(&self) -> TypeId {TypeId::of::<T>()}
    fn element_type_name(&self) -> &'static str {type_name::<T>()}

    fn swap_remove(&mut self, chunk: ChunkID, row: RowID) -> Result<Option<(ChunkID, RowID)>, AttributeError> {
        if chunk as usize >= self.chunks.len() {
            return Err(AttributeError::PositionOutOfBounds(PositionOutOfBoundsError {
                chunk, 
                row, 
                last_chunk_length: self.last_chunk_length,
                capacity: CHUNK_CAP,
                }));
            }

        let index = chunk as usize * CHUNK_CAP + row as usize;
        
        if index >= self.length {
            return Err(AttributeError::PositionOutOfBounds(PositionOutOfBoundsError {
                chunk,
                row,
                last_chunk_length: self.last_chunk_length,
                capacity: CHUNK_CAP,
                }));
            }


        let last_index = self.length - 1;
        let last_chunk = last_index / CHUNK_CAP;
        let last_row = last_index % CHUNK_CAP;


        unsafe {
            let removed = self.get_slot_unchecked(chunk as usize, row as usize).assume_init_read();
            let moved_from = if chunk as usize != last_chunk || row as usize != last_row {
                let last_value = self.get_slot_unchecked(last_chunk, last_row).assume_init_read();
                    self.get_slot_unchecked(chunk as usize, row as usize)
                    .write(MaybeUninit::new(last_value));
                    Some((last_chunk as ChunkID, last_row as RowID))
                } else {
                    None
            };

            drop(removed);
        }

        self.length -= 1;
        self.last_chunk_length = if self.last_chunk_length > 0 {
            self.last_chunk_length - 1
            } else {
            0
        };

        if self.last_chunk_length == 0 {
            self.chunks.pop();

            if !self.chunks.is_empty() {
                self.last_chunk_length = CHUNK_CAP;
            }
        }

        Ok(moved_from)
    }   

    fn push_dyn(&mut self, value: Box<dyn Any>) -> Result<(ChunkID, RowID), AttributeError> {
        if let Ok(v) = value.downcast::<T>() {
            return Attribute::<T>::push(self, *v);
        }
        let expected = TypeId::of::<T>();
        let actual = value.as_ref().type_id();
        Err(AttributeError::TypeMismatch(TypeMismatchError { expected, actual }))
    }

    fn push_from(
        &mut self, 
        source: &mut dyn TypeErasedAttribute, 
        source_chunk: ChunkID, 
        source_row: RowID
    ) ->Result<((ChunkID, RowID), Option<(ChunkID, RowID)>), AttributeError> {
        let source_attribute = source
            .as_any_mut()
            .downcast_mut::<Attribute<T>>()
            .expect("component type mismatch between attributes.");

        if !source_attribute.valid_position(source_chunk, source_row) {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk: source_chunk,
                row: source_row,
                chunks: source_attribute.chunks.len(),
                capacity: CHUNK_CAP,
                last_chunk_length: source_attribute.last_chunk_length,
            }));
        }

        let value = unsafe {
            source_attribute
                .get_slot_unchecked(source_chunk as usize, source_row as usize)
                .assume_init_read()
        };

        let (destination_chunk, destination_row) = Attribute::<T>::push(self, value).map_err(|e| {
            unsafe {
                source_attribute
                    .get_slot_unchecked(source_chunk as usize, source_row as usize)
                    .as_mut_ptr()
                    .write(value);
            }
            e
        })?;

        let source_index = source_chunk as usize * CHUNK_CAP + source_row as usize;
        let last_index = source_attribute.length - 1;
        let moved_from = if source_index != last_index {
            let last_chunk = last_index / CHUNK_CAP;
            let last_row = last_index % CHUNK_CAP;
            let last_value = unsafe {
                source_attribute.get_slot_unchecked(last_chunk, last_row).assume_init_read()
            };
            unsafe {
                source_attribute.get_slot_unchecked(source_chunk as usize, source_row as usize)
                          .write(MaybeUninit::new(last_value));
            }
            Some((last_chunk as ChunkID, last_row as RowID))
        } else {
            None
        };

        source_attribute.length -= 1;
        source_attribute.last_chunk_length = if source_attribute.last_chunk_length > 0 {
            source_attribute.last_chunk_length - 1
        } else {
            0
        };
        if source_attribute.last_chunk_length == 0 {
            source_attribute.chunks.pop();
            if !source_attribute.chunks.is_empty() {
                source_attribute.last_chunk_length = CHUNK_CAP;
            }
        }

        Ok(((destination_chunk, destination_row), moved_from))
    }
}

impl<T> Drop for Attribute<T> {
    fn drop(&mut self) {
        self.drop_all_initialized_elements();
    }
}
