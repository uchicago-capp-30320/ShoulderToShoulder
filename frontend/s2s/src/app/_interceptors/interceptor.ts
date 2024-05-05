import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';

// services
import { ApiService } from '../_services/api.service';

/**
 * Interceptor for adding the application token to HTTP requests.
 * 
 * This interceptor adds the application token to the headers of all HTTP requests.
 * 
 * @see ApiService
 */
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
    constructor(
        private apiService: ApiService
    ) {}

    intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        let authToken = this.apiService.appToken; // Assume a service that handles token retrieval
        let header = "X-App-Token"
        let content_type = "Content-Type"
        let content_type_value = "application/json"

        if (request.url.includes('profile') || request.url.includes('change_password') || request.url.includes('/user/')) {
            if (localStorage.getItem('access_token')) {
                authToken = `Bearer ${localStorage.getItem('access_token') as string}`;
                header = "Authorization";
            }
        }

        let authReq = request.clone({
            headers: request.headers.set(header, authToken).set(content_type, content_type_value),
        });

        if (request.url.includes('upload')) {
            authReq = request.clone({
                headers: request.headers.set(header, authToken),
            });
        }
        
        return next.handle(authReq);
    }
}
