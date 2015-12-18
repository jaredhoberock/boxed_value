// Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#ifndef __host__
#  define __host__
#  define BOXED_VALUE_UNDEF_HOST
#endif

#ifndef __device__
#  define __device__
#  define BOXED_VALUE_UNDEF_DEVICE
#endif

#include <memory>

template<class T, class Alloc = std::allocator<T>>
class boxed_value
{
  public:
    using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
    using value_type = typename allocator_type::value_type;

    __host__ __device__
    boxed_value()
      : boxed_value(value_type{})
    {}

    __host__ __device__
    boxed_value(const boxed_value& other)
      : boxed_value(other.value())
    {}

    __host__ __device__
    boxed_value(boxed_value&& other)
      : boxed_value(std::move(other.value()))
    {}

    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args&&...>::value
             >::type>
    __host__ __device__
    explicit boxed_value(Args... args)
      : data_(std::allocate_unique<T>(allocator_type(), std::forward<Args>(args)...))
    {}

    __host__ __device__
    boxed_value& operator=(const boxed_value& other)
    {
      value() = other.value();
      return *this;
    }

    __host__ __device__
    boxed_value& operator=(boxed_value&& other)
    {
      value() = std::move(other.value());
      return *this;
    }

    __host__ __device__
    value_type& value() &
    {
      return *data_;
    }

    __host__ __device__
    const value_type& value() const &
    {
      return *data_;
    }

    __host__ __device__
    value_type&& value() &&
    {
      return std::move(*data_);
    }

    __host__ __device__
    const value_type&& value() const &&
    {
      return std::move(*data_);
    }

    template<class U,
             class = typename std::enable_if<
               std::is_assignable<value_type,U&&>::value
             >::type>
    __host__ __device__
    boxed_value& operator=(U&& other)
    {
      value() = std::forward<U>(other);
      return *this;
    }

  private:
    struct deleter
    {
      __host__ __device__
      void operator()(T* ptr) const
      {
        // XXX should use allocator_traits::destroy()
        ptr->~T();

        // deallocate
        allocator_type alloc;
        alloc.deallocate(ptr, 1);
      }
    };

    // XXX ought to eliminate unique_ptr if we're going to allow this to be __host__ __device__
    std::unique_ptr<T,deleter> data_;
};


// when the allocator is std::allocator<T>, we can just put this on the stack
template<class T, class OtherT>
class boxed_value<T,std::allocator<OtherT>>
{
  public:
    using allocator_type = std::allocator<T>;
    using value_type = typename allocator_type::value_type;

    __host__ __device__
    boxed_value()
      : boxed_value(value_type{})
    {}

    __host__ __device__
    boxed_value(const boxed_value& other)
      : boxed_value(other.value())
    {}

    __host__ __device__
    boxed_value(boxed_value&& other)
      : boxed_value(std::move(other.value_))
    {}

    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args&&...>::value
             >::type>
    __host__ __device__
    explicit boxed_value(Args&&... args)
      : value_(std::forward<Args>(args)...)
    {}

    __host__ __device__
    value_type& value()
    {
      return value_;
    }

    __host__ __device__
    const value_type& value() const
    {
      return value_;
    }

    template<class U,
             class = typename std::enable_if<
               std::is_assignable<value_type,U&&>::value
             >::type>
    __host__ __device__
    boxed_value& operator=(U&& other)
    {
      value() = std::forward<U>(other);
      return *this;
    }

  private:
    value_type value_;
};


template<class T, class Alloc, class... Args>
__host__ __device__
boxed_value<T,Alloc> allocate_boxed(const Alloc&, Args&&... args)
{
  return boxed_value<T,Alloc>(std::forward<Args>(args)...);
}

#ifdef BOXED_VALUE_UNDEF_HOST
#  undef __host__
#  undef BOXED_VALUE_UNDEF_HOST
#endif

#ifdef BOXED_VALUE_UNDEF_DEVICE
#  undef __device__
#  undef BOXED_VALUE_UNDEF_DEVICE
#endif

