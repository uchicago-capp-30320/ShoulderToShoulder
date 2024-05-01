import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NgxMaskDirective, IConfig, provideEnvironmentNgxMask } from 'ngx-mask';
import {MatTooltipModule} from '@angular/material/tooltip';
import { MatButtonModule } from '@angular/material/button'; 
import { HashLocationStrategy, LocationStrategy } from '@angular/common';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { HTTP_INTERCEPTORS } from '@angular/common/http';
import { HttpClientModule } from '@angular/common/http';
const maskConfig: Partial<IConfig> = {
  validation: false,
};

// primeng
import { MenubarModule } from 'primeng/menubar';
import { BadgeModule } from 'primeng/badge';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { PasswordModule } from 'primeng/password';
import { CardModule } from 'primeng/card';
import { DividerModule } from 'primeng/divider';
import { DropdownModule } from 'primeng/dropdown';
import { MultiSelectModule } from 'primeng/multiselect';
import { TooltipModule } from 'primeng/tooltip';
import { ToggleButtonModule } from 'primeng/togglebutton';
import { RadioButtonModule } from 'primeng/radiobutton';
import { CheckboxModule } from 'primeng/checkbox';
import { DialogModule } from 'primeng/dialog';
import { ProgressBarModule } from 'primeng/progressbar';
import { ToastModule } from 'primeng/toast';
import { TableModule } from 'primeng/table';
import { FileUploadModule } from 'primeng/fileupload';

// Routing
import { AppRoutingModule } from './app-routing.module';

// Components
import { AppComponent } from './app.component';
import { HomepageComponent } from './homepage/homepage.component';
import { NavbarComponent } from './navbar/navbar.component';
import { AppHeaderComponent } from './app-header/app-header.component';
import { FooterComponent } from './footer/footer.component';
import { SignupPageComponent } from './signup-page/signup-page.component';
import { OnboardingComponent } from './onboarding/onboarding.component';
import { DemographicsSurveyComponent } from './demographics-survey/demographics-survey.component';
import { PreferencesSurveyComponent } from './preferences-survey/preferences-survey.component';
import { ScenariosSurveyComponent } from './scenarios-survey/scenarios-survey.component';
import { EventAvailabilitySurveyComponent } from './event-availability-survey/event-availability-survey.component';
import { LoaderComponent } from './loader/loader.component';
import { ProgressIndicatorComponent } from './progress-indicator/progress-indicator.component';

// HTTP interceptor
import { AuthInterceptor } from './_helpers/interceptor';
import { LogInComponent } from './log-in/log-in.component';
import { AvailabilityDisplayComponent } from './availability-display/availability-display.component';
import { ProfileComponent } from './profile/profile.component';
import { ProfileOverviewComponent } from './profile-overview/profile-overview.component';
import { ProfileSettingsComponent } from './profile-settings/profile-settings.component';

@NgModule({
  declarations: [
    AppComponent,
    HomepageComponent,
    NavbarComponent,
    AppHeaderComponent,
    FooterComponent,
    SignupPageComponent,
    OnboardingComponent,
    DemographicsSurveyComponent,
    PreferencesSurveyComponent,
    ScenariosSurveyComponent,
    EventAvailabilitySurveyComponent,
    LoaderComponent,
    ProgressIndicatorComponent,
    LogInComponent,
    AvailabilityDisplayComponent,
    ProfileComponent,
    ProfileOverviewComponent,
    ProfileSettingsComponent,
  ],
  imports: [
    BrowserModule,
    FormsModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    MenubarModule,
    BadgeModule,
    ButtonModule,
    InputTextModule,
    PasswordModule,
    CardModule,
    ReactiveFormsModule,
    NgxMaskDirective,
    MatTooltipModule,
    MatButtonModule,
    TooltipModule,
    DividerModule,
    DropdownModule,
    MultiSelectModule,
    ToggleButtonModule,
    HttpClientModule,
    RadioButtonModule,
    CheckboxModule,
    DialogModule,
    ProgressBarModule,
    ToastModule,
    TableModule,
    FileUploadModule
  ],
  providers: [
    provideEnvironmentNgxMask(maskConfig),
    provideAnimationsAsync(),
    { provide: LocationStrategy, useClass: HashLocationStrategy },
    { provide: HTTP_INTERCEPTORS, useClass: AuthInterceptor, multi: true }
    ],
  bootstrap: [AppComponent]
})
export class AppModule { }
