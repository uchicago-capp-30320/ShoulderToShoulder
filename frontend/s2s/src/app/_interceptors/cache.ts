import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';
import { withCache } from '@ngneat/cashew';

/**
 * Interceptor for caching certain HTTP requests.
 * 
 * This interceptor caches HTTP requests that are marked with the `withCache` operator.
 * 
 * @see ApiService
 */
@Injectable()
export class CacheInterceptor implements HttpInterceptor {
    constructor(
    ) {}

    intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        if (request.url.includes('calendar') || request.url.includes('availability')) {
            // calendar data should be cached for much longer than availability data
            let cacheTime = request.url.includes('calendar') ? 6.048e+8 : 300000;
            const cacheReq = request.clone({
                context: withCache({ ttl: cacheTime }) // Cache for 5 minutes,
            });
            
            return next.handle(cacheReq);
        }
        return next.handle(request);
    }
}
