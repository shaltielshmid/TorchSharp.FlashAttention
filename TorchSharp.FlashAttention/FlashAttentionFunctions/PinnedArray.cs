using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace TorchSharp.FlashAttention.FlashAttentionFunctions
{
    /// <summary>
    /// Allocator of T[] that pins the memory and handles unpinning.
    /// (taken from StackOverflow)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal sealed class PinnedArray<T> : IDisposable where T : struct
    {
        private GCHandle handle;

        public T[]? Array { get; private set; }

        public nint CreateArray(int length)
        {
            FreeHandle();

            Array = new T[length];

            // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
            try
            {
            }
            finally
            {
                handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
            }

            return handle.AddrOfPinnedObject();
        }

        public nint CreateArray(nint length)
        {
            return CreateArray((int)length);
        }

        public nint CreateArray(T[] array)
        {
            FreeHandle();

            Array = array;

            // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
            try
            {
            }
            finally
            {
                handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
            }

            return handle.AddrOfPinnedObject();
        }

        public void Dispose()
        {
            if (Array != null)
            {
                foreach (var val in Array)
                {
                    (val as IDisposable)?.Dispose();
                }
            }
            FreeHandle();
        }

        ~PinnedArray()
        {
            FreeHandle();
        }

        private void FreeHandle()
        {
            if (handle.IsAllocated)
            {
                handle.Free();
            }
        }
    }
}